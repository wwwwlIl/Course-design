"""
训练脚本
3D U-Net 脑部MRI肿瘤分割
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import UNet3D, CombinedLoss, calculate_dice_score
from dataset import create_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='3D U-Net BraTS Tumor Segmentation')

    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\24837\Desktop\Course design\data',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--patch_size', type=tuple, default=(128, 128, 128), help='裁剪尺寸')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')

    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    dice_scores = [[], [], []]

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, sample in enumerate(pbar):
        images = sample['image'].to(device)
        masks = sample['mask'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            batch_dice = calculate_dice_score(outputs, masks)
            for i in range(3):
                dice_scores[i].append(batch_dice[i])

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ET': f'{batch_dice[0]:.4f}',
            'WT': f'{batch_dice[1]:.4f}',
            'TC': f'{batch_dice[2]:.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_dice = [np.mean(scores) for scores in dice_scores]
    return avg_loss, avg_dice


def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    dice_scores = [[], [], []]

    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, sample in enumerate(pbar):
            images = sample['image'].to(device)
            masks = sample['mask'].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            batch_dice = calculate_dice_score(outputs, masks)
            for i in range(3):
                dice_scores[i].append(batch_dice[i])

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ET': f'{batch_dice[0]:.4f}',
                'WT': f'{batch_dice[1]:.4f}',
                'TC': f'{batch_dice[2]:.4f}'
            })

    avg_loss = total_loss / len(loader)
    avg_dice = [np.mean(scores) for scores in dice_scores]
    return avg_loss, avg_dice


def save_checkpoint(model, optimizer, epoch, dice, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dice': dice,
    }, save_path)
    print(f"模型已保存: {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("3D U-Net 脑部MRI肿瘤分割训练")
    print("=" * 60)

    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size
    )

    model = UNet3D(n_channels=4, n_classes=4).to(args.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_dice = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, args.device, epoch
        )

        mean_dice = np.mean(val_dice)
        scheduler.step(mean_dice)

        print(f"\n训练 Loss: {train_loss:.4f}")
        print(f"验证 Loss: {val_loss:.4f}")
        print(f"验证 Dice - ET: {val_dice[0]:.4f}, WT: {val_dice[1]:.4f}, TC: {val_dice[2]:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(model, optimizer, epoch, val_dice,
                            os.path.join(args.save_dir, 'best_model.pth'))

        save_checkpoint(model, optimizer, epoch, val_dice,
                        os.path.join(args.save_dir, 'latest_model.pth'))

    elapsed = time.time() - start_time
    print(f"\n训练完成! 总耗时: {elapsed / 3600:.2f} 小时")
    print(f"最佳验证Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()