"""
PyQt5 人机交互界面
脑部MRI肿瘤分割系统
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')  # ✅ 忽略所有警告

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Qt5Agg')

# ✅ 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QGroupBox, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from model import UNet3D


class PredictionThread(QThread):
    """后台预测线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, model_path, case_path, case_name, device):
        super().__init__()
        self.model_path = model_path
        self.case_path = case_path
        self.case_name = case_name
        self.device = device

    def run(self):
        try:
            self.progress.emit(10)

            # 加载模型
            model = UNet3D(n_channels=4, n_classes=4).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.progress.emit(30)

            # 加载数据
            modalities = ['flair', 't1', 't1ce', 't2']
            images = []
            for modality in modalities:
                file_path = os.path.join(self.case_path, f"{self.case_name}_{modality}.nii.gz")
                img = nib.load(file_path).get_fdata()
                images.append(img)

            image = np.stack(images, axis=0)

            # 归一化
            for i in range(image.shape[0]):
                modality = image[i]
                m = modality > 0
                if m.sum() > 0:
                    mean = modality[m].mean()
                    std = modality[m].std()
                    image[i][m] = (modality[m] - mean) / (std + 1e-8)

            self.progress.emit(50)

            # 滑动窗口预测
            _, h, w, d = image.shape
            patch_size = (128, 128, 128)
            ph, pw, pd = patch_size
            overlap = 0.5

            step_h = int(ph * (1 - overlap))
            step_w = int(pw * (1 - overlap))
            step_d = int(pd * (1 - overlap))

            output = np.zeros((4, h, w, d), dtype=np.float32)
            count = np.zeros((h, w, d), dtype=np.float32)

            total = ((h - ph) // step_h + 1) * ((w - pw) // step_w + 1) * ((d - pd) // step_d + 1)
            current = 0

            with torch.no_grad():
                for i in range(0, h - ph + 1, step_h):
                    for j in range(0, w - pw + 1, step_w):
                        for k in range(0, d - pd + 1, step_d):
                            patch = image[:, i:i+ph, j:j+pw, k:k+pd]
                            patch = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)
                            pred = model(patch)
                            pred = torch.softmax(pred, dim=1).squeeze().cpu().numpy()

                            output[:, i:i+ph, j:j+pw, k:k+pd] += pred
                            count[i:i+ph, j:j+pw, k:k+pd] += 1

                            current += 1
                            self.progress.emit(50 + int(40 * current / total))

            count[count == 0] = 1
            output /= count

            self.progress.emit(100)
            self.finished.emit(image, output)

        except Exception as e:
            self.error.emit(str(e))


class MriCanvas(FigureCanvas):
    """MRI图像显示画布"""

    def __init__(self, parent=None):
        self.fig, self.axes = plt.subplots(3, 6, figsize=(18, 9))
        super().__init__(self.fig)
        self.setParent(parent)
        self.colors = ['black', 'red', 'green', 'yellow']
        self.cmap = ListedColormap(self.colors)
        self.modality_names = ['FLAIR', 'T1', 'T1ce', 'T2']
        self.clear()

    def clear(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
        self.draw()

    def plot_results(self, image, pred, slice_indices):
        self.clear()
        pred_mask = np.argmax(pred, axis=0)

        for row, slice_idx in enumerate(slice_indices):
            for col in range(4):
                ax = self.axes[row, col]
                ax.imshow(image[col, :, :, slice_idx], cmap='gray')
                ax.set_title(f'{self.modality_names[col]}', fontsize=10)
                ax.axis('off')

            ax = self.axes[row, 5]
            ax.imshow(pred_mask[:, :, slice_idx], cmap=self.cmap, vmin=0, vmax=3)
            ax.set_title('AI预测', fontsize=10, color='blue')
            ax.axis('off')

        plt.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = None
        self.case_path = None
        self.case_name = None
        self.current_image = None
        self.current_pred = None
        self.slice_idx = 77

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('🧠 脑部MRI肿瘤分割系统')
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ========== 左侧控制面板 ==========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        title = QLabel('🧠 脑部MRI肿瘤分割系统')
        title.setFont(QFont('微软雅黑', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(line)

        # 1. 模型选择
        model_group = QGroupBox('📦 模型设置')
        model_layout = QVBoxLayout(model_group)

        self.model_label = QLabel('未选择模型')
        self.model_label.setStyleSheet('color: gray')
        model_layout.addWidget(self.model_label)

        btn_load_model = QPushButton('选择模型文件')
        btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(btn_load_model)
        left_layout.addWidget(model_group)

        # 2. 数据选择
        data_group = QGroupBox('📂 数据选择')
        data_layout = QVBoxLayout(data_group)

        self.case_label = QLabel('未选择病例')
        self.case_label.setStyleSheet('color: gray')
        data_layout.addWidget(self.case_label)

        btn_load_case = QPushButton('选择病例文件夹')
        btn_load_case.clicked.connect(self.load_case)
        data_layout.addWidget(btn_load_case)
        left_layout.addWidget(data_group)

        # 3. 设备信息
        device_group = QGroupBox('⚙️ 设备信息')
        device_layout = QVBoxLayout(device_group)

        device_text = f'当前设备: {"🟢 GPU" if self.device == "cuda" else "🟡 CPU"}'
        device_label = QLabel(device_text)
        device_layout.addWidget(device_label)
        left_layout.addWidget(device_group)

        # 4. 开始预测
        btn_predict = QPushButton('🚀 开始分割')
        btn_predict.setFont(QFont('微软雅黑', 12, QFont.Bold))
        btn_predict.setStyleSheet('''
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        ''')
        btn_predict.clicked.connect(self.start_prediction)
        left_layout.addWidget(btn_predict)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        # 5. 切片选择
        slice_group = QGroupBox('🎚️ 选择切片 (拖动滑块)')
        slice_layout = QVBoxLayout(slice_group)

        from PyQt5.QtWidgets import QSlider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 154)
        self.slice_slider.setValue(77)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_slider)

        self.slice_label = QLabel(f'当前切片: 77')
        self.slice_label.setAlignment(Qt.AlignCenter)
        slice_layout.addWidget(self.slice_label)
        left_layout.addWidget(slice_group)

        # 图例
        legend_group = QGroupBox('🎨 颜色图例')
        legend_layout = QVBoxLayout(legend_group)

        legends = [
            ('⚫ 黑色', '背景'),
            ('🔴 红色', '坏死/非增强肿瘤'),
            ('🟢 绿色', '水肿'),
            ('🟡 黄色', '增强肿瘤')
        ]

        for color, desc in legends:
            legend_layout.addWidget(QLabel(f'{color}: {desc}'))
        left_layout.addWidget(legend_group)

        left_layout.addStretch()

        # ========== 右侧显示区域 ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        display_title = QLabel('📊 分割结果可视化')
        display_title.setFont(QFont('微软雅黑', 14, QFont.Bold))
        display_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(display_title)

        self.canvas = MriCanvas()
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        self.statusBar().showMessage('✅ 系统就绪，请选择模型和数据')

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型文件', '', 'PyTorch模型 (*.pth)'
        )
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))
            self.model_label.setStyleSheet('color: green')
            self.statusBar().showMessage(f'✅ 模型已加载: {os.path.basename(path)}')

    def load_case(self):
        path = QFileDialog.getExistingDirectory(self, '选择病例文件夹')
        if path:
            self.case_path = path
            self.case_name = os.path.basename(path)
            self.case_label.setText(self.case_name)
            self.case_label.setStyleSheet('color: green')
            self.statusBar().showMessage(f'✅ 病例已加载: {self.case_name}')

    def start_prediction(self):
        if not self.model_path or not self.case_path:
            QMessageBox.warning(self, '警告', '请先选择模型和病例！')
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.thread = PredictionThread(
            self.model_path, self.case_path, self.case_name, self.device
        )
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_prediction_finished)
        self.thread.error.connect(self.on_prediction_error)
        self.thread.start()

        self.statusBar().showMessage('⏳ 正在进行肿瘤分割...')

    def on_prediction_finished(self, image, pred):
        self.current_image = image
        self.current_pred = pred
        self.update_display()
        self.progress.setVisible(False)
        self.statusBar().showMessage('✅ 分割完成！拖动滑块查看不同切片')

        QMessageBox.information(self, '成功', '肿瘤分割完成！\n\n拖动滑块查看不同切片的结果')

    def on_prediction_error(self, error_msg):
        self.progress.setVisible(False)
        QMessageBox.critical(self, '错误', f'预测失败:\n{error_msg}')
        self.statusBar().showMessage('❌ 分割失败')

    def update_slice(self):
        self.slice_idx = self.slice_slider.value()
        self.slice_label.setText(f'当前切片: {self.slice_idx}')
        self.update_display()

    def update_display(self):
        if self.current_image is not None and self.current_pred is not None:
            slices = [max(0, self.slice_idx-1), self.slice_idx, min(154, self.slice_idx+1)]
            self.canvas.plot_results(self.current_image, self.current_pred, slices)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('微软雅黑', 9))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()