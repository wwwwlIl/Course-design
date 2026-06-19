"""
预测与可视化脚本
对脑部MRI进行肿瘤分割并可视化结果
"""

import os
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from model import UNet3D


def parse_args():
    parser = argparse.ArgumentParser(description='BraTS Tumor Segmentation Prediction')

    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\24837\Desktop\Course design\data',
                        help='数据集路径')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--case_id', type=str, default='BraTS2021_00000',
                        help='预测病例ID')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='结果保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='预测设备')

    return parser.parse_args()


def load_case(case_path, case_name):
    """加载单个病例的4个模态"""
    modalities = ['flair', 't1', 't1ce', 't2']
    images = []

    for modality in modalities:
        file_path = os.path.join(case_path, f"{case_name}_{modality}.nii.gz")
        img = nib.load(file_path).get_fdata()
        images.append(img)

    seg_path = os.path.join(case_path, f"{case_name}_seg.nii.gz")
    mask = None
    if os.path.exists(seg_path):
        mask = nib.load(seg_path).get_fdata()
        mask[mask == 4] = 3

    image = np.stack(images, axis=0)

    for i in range(image.shape[0]):
        modality = image[i]
        m = modality > 0
        if m.sum() > 0:
            mean = modality[m].mean()
            std = modality[m].std()
            image[i][m] = (modality[m] - mean) / (std + 1e-8)

    return image, mask


def predict_sliding_window(model, image, device, patch_size=(128, 128, 128), overlap=0.5):
    """滑动窗口预测整图"""
    _, h, w, d = image.shape
    ph, pw, pd = patch_size

    step_h = int(ph * (1 - overlap))
    step_w = int(pw * (1 - overlap))
    step_d = int(pd * (1 - overlap))

    output = np.zeros((4, h, w, d), dtype=np.float32)
    count = np.zeros((h, w, d), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, h - ph + 1, step_h):
            for j in range(0, w - pw + 1, step_w):
                for k in range(0, d - pd + 1, step_d):
                    patch = image[:, i:i+ph, j:j+pw, k:k+pd]
                    patch = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                    pred = model(patch)
                    pred = torch.softmax(pred, dim=1).squeeze().cpu().numpy()

                    output[:, i:i+ph, j:j+pw, k:k+pd] += pred
                    count[i:i+ph, j:j+pw, k:k+pd] += 1

    count[count == 0] = 1
    output /= count

    return output


def visualize_results(image, mask, pred, save_path, case_name):
    """可视化分割结果"""
    colors = ['black', 'red', 'green', 'yellow']
    cmap = ListedColormap(colors)

    _, h, w, d = image.shape

    if mask is not None:
        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    else:
        tumor_slices = np.where(pred.sum(axis=(0, 1, 2)) > 0)[0]

    if len(tumor_slices) > 0:
        mid = len(tumor_slices) // 2
        slices = tumor_slices[[max(0, mid-1), mid, min(len(tumor_slices)-1, mid+1)]]
    else:
        slices = [d//4, d//2, 3*d//4]

    n_rows = len(slices)
    n_cols = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows))
    modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']

    for row, slice_idx in enumerate(slices):
        for col in range(4):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            img = image[col, :, :, slice_idx]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{modality_names[col]} (z={slice_idx})')
            ax.axis('off')

        ax = axes[row, 4] if n_rows > 1 else axes[4]
        if mask is not None:
            ax.imshow(mask[:, :, slice_idx], cmap=cmap, vmin=0, vmax=3)
            ax.set_title('Ground Truth')
        else:
            ax.text(0.5, 0.5, 'No GT', ha='center', va='center')
            ax.set_title('Ground Truth')
        ax.axis('off')

        ax = axes[row, 5] if n_rows > 1 else axes[5]
        pred_slice = np.argmax(pred, axis=0)[:, :, slice_idx]
        ax.imshow(pred_slice, cmap=cmap, vmin=0, vmax=3)
        ax.set_title('Prediction')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{case_name}_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存: {os.path.join(save_path, f'{case_name}_visualization.png')}")


def save_prediction_nifti(pred, affine, save_path, case_name):
    """保存预测结果为NIfTI格式"""
    pred_mask = np.argmax(pred, axis=0)
    pred_mask[pred_mask == 3] = 4

    img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine)
    nib.save(img, os.path.join(save_path, f'{case_name}_prediction.nii.gz'))
    print(f"分割结果已保存: {os.path.join(save_path, f'{case_name}_prediction.nii.gz')}")


def calculate_metrics(pred, mask):
    """计算评估指标"""
    if mask is None:
        return None

    pred_mask = np.argmax(pred, axis=0)

    dice_scores = []
    class_names = ['NCR/NET (1)', 'ED (2)', 'ET (3)']

    for class_idx in range(1, 4):
        pred_bin = (pred_mask == class_idx)
        mask_bin = (mask == class_idx)

        intersection = np.logical_and(pred_bin, mask_bin).sum()
        union = pred_bin.sum() + mask_bin.sum()

        dice = (2. * intersection) / (union + 1e-8)
        dice_scores.append(dice)
        print(f"{class_names[class_idx-1]} Dice: {dice:.4f}")

    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    return dice_scores


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("脑部MRI肿瘤分割预测")
    print("=" * 60)
    print(f"病例: {args.case_id}")
    print(f"模型: {args.model_path}")
    print(f"设备: {args.device}")
    print("=" * 60)

    print("\n加载模型...")
    model = UNet3D(n_channels=4, n_classes=4).to(args.device)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功! (Epoch {checkpoint['epoch']})")
    else:
        print("警告: 未找到预训练模型，使用随机初始化!")

    print(f"\n加载病例: {args.case_id}")
    case_path = os.path.join(args.data_dir, args.case_id)
    image, mask = load_case(case_path, args.case_id)
    print(f"图像形状: {image.shape}")

    flair_path = os.path.join(case_path, f"{args.case_id}_flair.nii.gz")
    affine = nib.load(flair_path).affine

    print("\n开始预测...")
    pred = predict_sliding_window(model, image, args.device)
    print(f"预测完成!")

    if mask is not None:
        print("\n评估指标:")
        calculate_metrics(pred, mask)

    print("\n生成可视化...")
    visualize_results(image, mask, pred, args.save_dir, args.case_id)

    # ✅ 现在会保存NIfTI了！
    save_prediction_nifti(pred, affine, args.save_dir, args.case_id)

    print("\n" + "=" * 60)
    print("预测完成!")
    print(f"结果保存在: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()