#!/usr/bin/env python3
"""
GSSE图形化界面 (Gaussian Splatting Semantic Editor)

"""

import os
import sys

def setup_qt_plugin_path():
    """设置Qt插件路径，排除OpenCV的插件路径"""
    import site
    
    # 1. 查找PyQt5的正确插件路径
    pyqt5_plugin_path = None
    
    # 方法1: 通过sys.path查找
    for path in sys.path:
        if 'site-packages' in path:
            possible_paths = [
                os.path.join(path, 'PyQt5', 'Qt5', 'plugins'),
                os.path.join(path, 'PyQt5', 'Qt', 'plugins'),
            ]
            for plugin_path in possible_paths:
                xcb_plugin = os.path.join(plugin_path, 'platforms', 'libqxcb.so')
                if os.path.exists(plugin_path) and os.path.exists(xcb_plugin):
                    pyqt5_plugin_path = plugin_path
                    break
            if pyqt5_plugin_path:
                break
    
    # 方法2: 通过conda环境直接查找
    if not pyqt5_plugin_path:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            # 直接查找conda环境的lib目录
            conda_lib = os.path.join(conda_prefix, 'lib')
            if os.path.exists(conda_lib):
                # 查找所有可能的Python版本目录
                for item in os.listdir(conda_lib):
                    if item.startswith('python'):
                        plugin_path = os.path.join(conda_lib, item, 'site-packages', 'PyQt5', 'Qt5', 'plugins')
                        xcb_plugin = os.path.join(plugin_path, 'platforms', 'libqxcb.so')
                        if os.path.exists(plugin_path) and os.path.exists(xcb_plugin):
                            pyqt5_plugin_path = plugin_path
                            break
                    if pyqt5_plugin_path:
                        break
                
                # 如果没找到，尝试直接查找PyQt5包
                if not pyqt5_plugin_path:
                    for item in os.listdir(conda_lib):
                        if item.startswith('python'):
                            pyqt5_base = os.path.join(conda_lib, item, 'site-packages', 'PyQt5')
                            if os.path.exists(pyqt5_base):
                                # 查找Qt5或Qt目录下的plugins
                                for qt_dir in ['Qt5', 'Qt']:
                                    plugin_path = os.path.join(pyqt5_base, qt_dir, 'plugins')
                                    xcb_plugin = os.path.join(plugin_path, 'platforms', 'libqxcb.so')
                                    if os.path.exists(plugin_path) and os.path.exists(xcb_plugin):
                                        pyqt5_plugin_path = plugin_path
                                        break
                                if pyqt5_plugin_path:
                                    break
                        if pyqt5_plugin_path:
                            break
    
    # 方法3: 通过site-packages查找（更直接的方式）
    if not pyqt5_plugin_path:
        try:
            import site
            site_packages = site.getsitepackages()
            for sp in site_packages:
                pyqt5_base = os.path.join(sp, 'PyQt5')
                if os.path.exists(pyqt5_base):
                    for qt_dir in ['Qt5', 'Qt']:
                        plugin_path = os.path.join(pyqt5_base, qt_dir, 'plugins')
                        xcb_plugin = os.path.join(plugin_path, 'platforms', 'libqxcb.so')
                        if os.path.exists(plugin_path) and os.path.exists(xcb_plugin):
                            pyqt5_plugin_path = plugin_path
                            break
                    if pyqt5_plugin_path:
                        break
        except:
            pass
    
    # 方法4: 通过conda的qt-main包查找（conda安装的Qt可能在这里）
    if not pyqt5_plugin_path:
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        if conda_prefix:
            # conda安装的qt-main包的plugins目录
            qt_plugin_paths = [
                os.path.join(conda_prefix, 'plugins'),
                os.path.join(conda_prefix, 'lib', 'plugins'),
            ]
            for plugin_path in qt_plugin_paths:
                xcb_plugin = os.path.join(plugin_path, 'platforms', 'libqxcb.so')
                if os.path.exists(plugin_path) and os.path.exists(xcb_plugin):
                    pyqt5_plugin_path = plugin_path
                    break
    
    # 2. 设置插件路径，排除OpenCV
    if pyqt5_plugin_path:
        # 只使用PyQt5的插件路径
        os.environ['QT_PLUGIN_PATH'] = pyqt5_plugin_path
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt5_plugin_path
        print(f"[DEBUG] 设置Qt插件路径: {pyqt5_plugin_path}", file=sys.stderr)
    else:
        # 如果找不到PyQt5插件路径，至少过滤掉OpenCV的路径
        current_paths = os.environ.get('QT_PLUGIN_PATH', '').split(os.pathsep)
        filtered_paths = [p for p in current_paths if p and 'cv2' not in p.lower()]
        if filtered_paths:
            os.environ['QT_PLUGIN_PATH'] = os.pathsep.join(filtered_paths)
        elif 'QT_PLUGIN_PATH' in os.environ:
            del os.environ['QT_PLUGIN_PATH']
        
        if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
            path = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
            if 'cv2' in path.lower():
                del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    
    # 3. 确保使用xcb平台
    if 'QT_QPA_PLATFORM' not in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

import argparse
import threading
import traceback
import gc
import subprocess
import tempfile
import shutil
import time
from typing import Optional, Tuple, List
from datetime import datetime

import torch
import numpy as np

# 导入cv2（它可能会设置Qt插件路径，但我们会在之后修复）
import cv2

# cv2导入后，立即修复Qt插件路径（必须在导入PyQt5之前）
# cv2可能会设置错误的插件路径，我们需要覆盖它
setup_qt_plugin_path()

from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, 
    QGroupBox, QTabWidget, QScrollArea, QSlider, QProgressBar,
    QListWidget, QSplitter, QFrame, QButtonGroup, QSizePolicy, QColorDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSize, QTimer, QUrl, QSettings
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QWheelEvent,
    QMouseEvent, QKeySequence, QIcon, QPalette, QDesktopServices
)

# 3D可视化相关导入（已移除matplotlib，仅保留SIBR查看器相关功能）

# 设置CUDA内存管理优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# GSSE imports
from seg_utils import grounding_dino_prompt
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.gaussian_renderer import render

# Import functions from run_sags.py
sys.path.append(os.path.dirname(__file__))
from run_gsse import (
    init_sam_predictor, text_prompting, self_prompt,
    generate_3d_prompts, porject_to_2d, mask_inverse, ensemble, gaussian_decomp, save_gs,
    _resize_mask_torch, init_fastsam_predictor, fastsam_point_prompt, fastsam_text_prompt,
    post_process_mask
)

# Import COLMAP processor
from colmap import (COLMAPProcessor, validate_image_directory, validate_video_file, 
                    get_video_info, convert_colmap_to_gaussian_splatting_format)

# Import SOG trainer
from sog import SOGTrainer

# Import SAGA module
try:
    from saga_module import SAGAModule
    SAGA_AVAILABLE = True
except ImportError:
    SAGA_AVAILABLE = False
    print("警告: SAGA模块不可用")

# Import Gaussian editor
try:
    from gaussian_editor import GaussianEditor, EditType, EditSelectOpType, UndoRedoSystem, HIDE_STATE, LOCK_STATE
    # 使用多模式渲染器
    from gaussian_multi_mode_renderer import GaussianMultiModeRenderer as Gaussian3DViewer, RenderMode
    MULTI_MODE_AVAILABLE = True
    EDITOR_AVAILABLE = True
except ImportError as e:
    EDITOR_AVAILABLE = False
    print(f"警告: 编辑模块不可用: {e}")

# Import GIS modules
try:
    from cesium_widget import CesiumWidget, CesiumPanel
    from gis_converter import (
        CoordinateTransformer, GaussianSplattingToGIS,
        GeoDataConfig, PRESET_LOCATIONS
    )
    GIS_AVAILABLE = True
except ImportError as e:
    GIS_AVAILABLE = False
    print(f"警告: GIS模块不可用: {e}")

# Import HTTP server
try:
    from temp_http_server import start_global_server, stop_global_server, get_global_server
    HTTP_SERVER_AVAILABLE = True
except ImportError as e:
    HTTP_SERVER_AVAILABLE = False
    print(f"警告: HTTP服务器模块不可用: {e}")


class MemoryManager:
    """GPU内存管理工具类"""
    
    @staticmethod
    def clear_gpu_memory():
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    @staticmethod
    def get_gpu_memory_info():
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return "CUDA不可用"
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return f"已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB, 总计: {total:.2f}GB"
    
    @staticmethod
    def safe_model_load(model_func, *args, **kwargs):
        """安全加载模型，支持CPU回退"""
        try:
            # 尝试GPU加载
            MemoryManager.clear_gpu_memory()
            return model_func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            print("GPU内存不足，尝试CPU回退...")
            MemoryManager.clear_gpu_memory()
            return model_func(*args, **kwargs)
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    @staticmethod
    def optimize_memory_settings():
        """优化内存设置"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 启用内存碎片整理
            torch.cuda.set_per_process_memory_fraction(0.9)


class ImageCanvas(QWidget):
    """支持缩放、拖拽的图像显示画布"""
    
    # 信号：点击事件
    clicked = pyqtSignal(int, int)  # x, y
    # 信号：设置图像（用于线程安全）
    # 注意：不能直接使用np.ndarray，使用object类型
    set_image_signal = pyqtSignal(object)  # image: np.ndarray
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.image = None
        self.pixmap = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_pos = QPoint()
        
        # 点击点列表 [(x, y, label), ...]
        self.click_points = []
        
        # 设置深色背景（使用全局主题）
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(37, 37, 37))  # #252525
        self.setPalette(palette)
        
        # 连接信号
        self.set_image_signal.connect(self._set_image_internal)
        
    def set_image(self, image: np.ndarray):
        """设置显示的图像（线程安全）"""
        # 使用信号确保在主线程中更新
        self.set_image_signal.emit(image)
    
    def _set_image_internal(self, image):
        """设置显示的图像（内部方法，在主线程中调用）"""
        if image is None:
            self.image = None
            self.pixmap = None
            self.update()
            return
        
        # 确保是numpy数组
        if not isinstance(image, np.ndarray):
            return
            
        # 转换为RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # 假设已经是RGB格式（PIL和PyTorch渲染结果都是RGB）
            # 如果图像确实是BGR格式，需要在调用set_image之前转换
            pass
        
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        
        # 确保数组是连续的，并转换为bytes
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        image_bytes = image.tobytes()
        
        q_image = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(q_image)
        self.image = image.copy()  # 创建副本以避免内存问题
        self.fit_to_window()
        self.update()
    
    def fit_to_window(self):
        """调整图像以适应窗口"""
        if self.pixmap is None:
            return
        
        w = self.width()
        h = self.height()
        if w == 0 or h == 0:
            return
        
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()
        
        if pix_w == 0 or pix_h == 0:
            return
        
        zoom_w = (w - 20) / pix_w
        zoom_h = (h - 20) / pix_h
        self.zoom = min(zoom_w, zoom_h, 10.0)
        self.zoom = max(0.2, self.zoom)
        
        scaled_w = pix_w * self.zoom
        scaled_h = pix_h * self.zoom
        self.offset_x = (w - scaled_w) / 2
        self.offset_y = (h - scaled_h) / 2
    
    def clear_points(self):
        """清除所有点击点"""
        self.click_points = []
        self.update()
    
    def add_point(self, x, y, label):
        """添加点击点"""
        # 将窗口坐标转换为图像坐标
        img_x = int((x - self.offset_x) / self.zoom)
        img_y = int((y - self.offset_y) / self.zoom)
        if self.image is not None:
            img_h, img_w = self.image.shape[:2]
            img_x = max(0, min(img_w - 1, img_x))
            img_y = max(0, min(img_h - 1, img_y))
            self.click_points.append((img_x, img_y, label))
            self.update()
    
    def remove_last_point(self):
        """移除最后一个点"""
        if self.click_points:
            self.click_points.pop()
            self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮缩放"""
        if self.pixmap is None:
            return
        
        delta = event.angleDelta().y()
        if delta == 0:
            return
        
        old_zoom = self.zoom
        if delta > 0:
            new_zoom = old_zoom * 1.1
        else:
            new_zoom = old_zoom / 1.1
        
        new_zoom = max(0.2, min(10.0, new_zoom))
        
        # 以鼠标位置为中心缩放
        pos = event.pos()
        cx, cy = pos.x(), pos.y()
        ox, oy = self.offset_x, self.offset_y
        
        scale = new_zoom / old_zoom
        self.offset_x = cx - scale * (cx - ox)
        self.offset_y = cy - scale * (cy - oy)
        self.zoom = new_zoom
        
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 左键点击添加点
            x, y = event.pos().x(), event.pos().y()
            # 默认为正点，可以通过外部控制
            self.clicked.emit(x, y)
        elif event.button() == Qt.RightButton or event.button() == Qt.MidButton:
            # 右键或中键开始拖拽
            self.dragging = True
            self.last_pos = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self.dragging:
            dx = event.pos().x() - self.last_pos.x()
            dy = event.pos().y() - self.last_pos.y()
            self.offset_x += dx
            self.offset_y += dy
            self.last_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.RightButton or event.button() == Qt.MidButton:
            self.dragging = False
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        if self.pixmap is None:
            super().resizeEvent(event)
            return
        
        # 如果图像刚设置，适应窗口
        if self.zoom == 1.0 and self.offset_x == 0 and self.offset_y == 0:
            self.fit_to_window()
        else:
            self.update()
        super().resizeEvent(event)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), Qt.white)
            
            if self.pixmap is None:
                return
            
            # 绘制图像
            scaled_pixmap = self.pixmap.scaled(
                int(self.pixmap.width() * self.zoom),
                int(self.pixmap.height() * self.zoom),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), scaled_pixmap)
            
            # 绘制点击点
            if self.image is not None:
                img_h, img_w = self.image.shape[:2]
                for px, py, label in self.click_points:
                    # 将图像坐标转换为窗口坐标
                    x = int(self.offset_x + px * self.zoom)
                    y = int(self.offset_y + py * self.zoom)
                    
                    # 绘制点
                    color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
                    painter.setPen(QPen(color, 3))
                    painter.setBrush(QColor(color))
                    radius = 8
                    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
                    
                    # 绘制标签
                    label_text = "+" if label == 1 else "-"
                    painter.setPen(Qt.white)
                    painter.setFont(QFont("Arial", 10, QFont.Bold))
                    painter.drawText(x - 5, y + 5, label_text)
        finally:
            painter.end()


class ThemeManager:
    """主题管理器"""
    
    @staticmethod
    def get_dark_theme():
        """获取Adobe风格深色主题"""
        return """
        /* Adobe风格深色主题 - 全局样式 */
        
        /* 主窗口和基础控件 */
        QMainWindow, QWidget {
            background-color: #323232;
            color: #E0E0E0;
        }
        
        /* 菜单栏 */
        QMenuBar {
            background-color: #2B2B2B;
            color: #E0E0E0;
            border-bottom: 1px solid #404040;
            padding: 2px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
            border-radius: 2px;
        }
        
        QMenuBar::item:selected {
            background-color: #404040;
        }
        
        QMenuBar::item:pressed {
            background-color: #505050;
        }
        
        /* 下拉菜单 */
        QMenu {
            background-color: #2B2B2B;
            color: #E0E0E0;
            border: 1px solid #404040;
            padding: 2px;
        }
        
        QMenu::item {
            padding: 4px 20px 4px 25px;
            border-radius: 2px;
        }
        
        QMenu::item:selected {
            background-color: #0E639C;
            color: white;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #404040;
            margin: 2px 5px;
        }
        
        /* 状态栏 */
        QStatusBar {
            background-color: #2B2B2B;
            color: #E0E0E0;
            border-top: 1px solid #404040;
        }
        
        /* 标签页 */
        QTabWidget::pane {
            border: 1px solid #404040;
            background-color: #323232;
            top: -1px;
        }
        
        QTabBar::tab {
            background-color: #2B2B2B;
            color: #B0B0B0;
            border: 1px solid #404040;
            border-bottom: none;
            padding: 6px 12px;
            margin-right: 2px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        
        QTabBar::tab:selected {
            background-color: #323232;
            color: #E0E0E0;
            border-bottom: 1px solid #323232;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #383838;
            color: #E0E0E0;
        }
        
        /* 按钮 */
        QPushButton {
            background-color: #424242;
            color: #E0E0E0;
            border: 1px solid #505050;
            border-radius: 3px;
            padding: 5px 12px;
            min-height: 22px;
            font-size: 16px;
        }
        
        QPushButton:hover {
            background-color: #505050;
            border-color: #606060;
        }
        
        QPushButton:pressed {
            background-color: #383838;
            border-color: #404040;
        }
        
        QPushButton:checked {
            background-color: #0E639C;
            color: white;
            border-color: #0E639C;
        }
        
        QPushButton:checked:hover {
            background-color: #1177BB;
        }
        
        QPushButton:disabled {
            background-color: #2B2B2B;
            color: #707070;
            border-color: #353535;
        }
        
        /* 输入框 */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #252525;
            color: #E0E0E0;
            border: 1px solid #404040;
            border-radius: 2px;
            padding: 4px;
            selection-background-color: #0E639C;
            selection-color: white;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #0E639C;
            background-color: #2B2B2B;
        }
        
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
            background-color: #2B2B2B;
            color: #707070;
            border-color: #353535;
        }
        
        /* 下拉框 */
        QComboBox {
            background-color: #252525;
            color: #E0E0E0;
            border: 1px solid #404040;
            border-radius: 2px;
            padding: 4px;
            min-height: 22px;
        }
        
        QComboBox:hover {
            border-color: #505050;
        }
        
        QComboBox:focus {
            border: 1px solid #0E639C;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #B0B0B0;
            width: 0;
            height: 0;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2B2B2B;
            color: #E0E0E0;
            border: 1px solid #404040;
            selection-background-color: #0E639C;
            selection-color: white;
        }
        
        QComboBox:disabled {
            background-color: #2B2B2B;
            color: #707070;
            border-color: #353535;
        }
        
        /* 复选框和单选按钮 */
        QCheckBox, QRadioButton {
            color: #E0E0E0;
            spacing: 5px;
        }
        
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #505050;
            background-color: #252525;
            border-radius: 2px;
        }
        
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {
            border-color: #606060;
            background-color: #2B2B2B;
        }
        
        QCheckBox::indicator:checked {
            background-color: #0E639C;
            border-color: #0E639C;
        }
        
        QRadioButton::indicator {
            border-radius: 8px;
        }
        
        QRadioButton::indicator:checked {
            background-color: #0E639C;
            border-color: #0E639C;
        }
        
        QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
            background-color: #2B2B2B;
            border-color: #353535;
        }
        
        /* 分组框 */
        QGroupBox {
            border: 1px solid #404040;
            border-radius: 3px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #2B2B2B;
            color: #E0E0E0;
            font-weight: bold;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            background-color: #2B2B2B;
            color: #E0E0E0;
        }
        
        /* 标签 */
        QLabel {
            color: #E0E0E0;
            background-color: transparent;
        }
        
        /* 滚动条 */
        QScrollBar:vertical {
            background-color: #2B2B2B;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #505050;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }
        
        QScrollBar::handle:vertical:pressed {
            background-color: #707070;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #2B2B2B;
            height: 12px;
            border: none;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #505050;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #606060;
        }
        
        QScrollBar::handle:horizontal:pressed {
            background-color: #707070;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* 滑块 */
        QSlider::groove:horizontal {
            background-color: #404040;
            height: 4px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0E639C;
            border: 1px solid #1177BB;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #1177BB;
        }
        
        QSlider::handle:horizontal:pressed {
            background-color: #0A4D75;
        }
        
        /* 进度条 */
        QProgressBar {
            background-color: #252525;
            border: 1px solid #404040;
            border-radius: 2px;
            text-align: center;
            color: #E0E0E0;
            height: 20px;
        }
        
        QProgressBar::chunk {
            background-color: #0E639C;
            border-radius: 1px;
        }
        
        /* 列表和表格 */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #252525;
            color: #E0E0E0;
            border: 1px solid #404040;
            selection-background-color: #0E639C;
            selection-color: white;
            alternate-background-color: #2B2B2B;
        }
        
        /* 分割器 */
        QSplitter::handle {
            background-color: #404040;
        }
        
        QSplitter::handle:horizontal {
            width: 1px;
        }
        
        QSplitter::handle:vertical {
            height: 1px;
        }
        
        QSplitter::handle:hover {
            background-color: #505050;
        }
        
        /* 滚动区域 */
        QScrollArea {
            background-color: #323232;
            border: none;
        }
        
        /* 工具提示 */
        QToolTip {
            background-color: #2B2B2B;
            color: #E0E0E0;
            border: 1px solid #404040;
            padding: 4px;
            border-radius: 2px;
        }
        
        /* 对话框 */
        QDialog {
            background-color: #323232;
            color: #E0E0E0;
        }
        
        QMessageBox {
            background-color: #323232;
            color: #E0E0E0;
        }
        
        QMessageBox QLabel {
            color: #E0E0E0;
        }
        
        /* 框架 */
        QFrame {
            background-color: transparent;
            color: #E0E0E0;
        }
        
        /* 旋钮框 */
        QSpinBox, QDoubleSpinBox {
            background-color: #252525;
            color: #E0E0E0;
            border: 1px solid #404040;
            border-radius: 2px;
            padding: 4px;
            min-height: 22px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #0E639C;
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            background-color: #424242;
            border-left: 1px solid #404040;
            border-top-right-radius: 2px;
        }
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
            background-color: #505050;
        }
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #424242;
            border-left: 1px solid #404040;
            border-bottom-right-radius: 2px;
        }
        
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #505050;
        }
        """
    
    @staticmethod
    def get_light_theme():
        """获取浅色主题"""
        return """
        /* 浅色主题 - 全局样式 */
        
        /* 主窗口和基础控件 */
        QMainWindow, QWidget {
            background-color: #F5F5F5;
            color: #212529;
        }
        
        /* 菜单栏 */
        QMenuBar {
            background-color: #FFFFFF;
            color: #212529;
            border-bottom: 1px solid #CCCCCC;
            padding: 2px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
            border-radius: 2px;
        }
        
        QMenuBar::item:selected {
            background-color: #E0E0E0;
        }
        
        QMenuBar::item:pressed {
            background-color: #D0D0D0;
        }
        
        /* 下拉菜单 */
        QMenu {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            padding: 2px;
        }
        
        QMenu::item {
            padding: 4px 20px 4px 25px;
            border-radius: 2px;
        }
        
        QMenu::item:selected {
            background-color: #2196F3;
            color: white;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #CCCCCC;
            margin: 2px 5px;
        }
        
        /* 状态栏 */
        QStatusBar {
            background-color: #FFFFFF;
            color: #212529;
            border-top: 1px solid #CCCCCC;
        }
        
        /* 标签页 */
        QTabWidget::pane {
            border: 1px solid #CCCCCC;
            background-color: #F5F5F5;
            top: -1px;
        }
        
        QTabBar::tab {
            background-color: #F0F0F0;
            color: #666666;
            border: 1px solid #CCCCCC;
            border-bottom: none;
            padding: 6px 12px;
            margin-right: 2px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }
        
        QTabBar::tab:selected {
            background-color: #F5F5F5;
            color: #212529;
            border-bottom: 1px solid #F5F5F5;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #F8F8F8;
            color: #212529;
        }
        
        /* 按钮 */
        QPushButton {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            padding: 5px 12px;
            min-height: 22px;
            font-size: 16px;
        }
        
        QPushButton:hover {
            background-color: #F0F0F0;
            border-color: #999999;
        }
        
        QPushButton:pressed {
            background-color: #E0E0E0;
            border-color: #CCCCCC;
        }
        
        QPushButton:checked {
            background-color: #2196F3;
            color: white;
            border-color: #2196F3;
        }
        
        QPushButton:checked:hover {
            background-color: #1976D2;
        }
        
        QPushButton:disabled {
            background-color: #F0F0F0;
            color: #999999;
            border-color: #E0E0E0;
        }
        
        /* 输入框 */
        QLineEdit, QTextEdit, QPlainTextEdit {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            border-radius: 2px;
            padding: 4px;
            selection-background-color: #2196F3;
            selection-color: white;
        }
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border: 1px solid #2196F3;
            background-color: #FFFFFF;
        }
        
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
            background-color: #F0F0F0;
            color: #999999;
            border-color: #E0E0E0;
        }
        
        /* 下拉框 */
        QComboBox {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            border-radius: 2px;
            padding: 4px;
            min-height: 22px;
        }
        
        QComboBox:hover {
            border-color: #999999;
        }
        
        QComboBox:focus {
            border: 1px solid #2196F3;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #666666;
            width: 0;
            height: 0;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            selection-background-color: #2196F3;
            selection-color: white;
        }
        
        QComboBox:disabled {
            background-color: #F0F0F0;
            color: #999999;
            border-color: #E0E0E0;
        }
        
        /* 复选框和单选按钮 */
        QCheckBox, QRadioButton {
            color: #212529;
            spacing: 5px;
        }
        
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #CCCCCC;
            background-color: #FFFFFF;
            border-radius: 2px;
        }
        
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {
            border-color: #999999;
            background-color: #F8F8F8;
        }
        
        QCheckBox::indicator:checked {
            background-color: #2196F3;
            border-color: #2196F3;
        }
        
        QRadioButton::indicator {
            border-radius: 8px;
        }
        
        QRadioButton::indicator:checked {
            background-color: #2196F3;
            border-color: #2196F3;
        }
        
        QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
            background-color: #F0F0F0;
            border-color: #E0E0E0;
        }
        
        /* 分组框 */
        QGroupBox {
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #F9F9F9;
            color: #212529;
            font-weight: bold;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            background-color: #F9F9F9;
            color: #212529;
        }
        
        /* 标签 */
        QLabel {
            color: #212529;
            background-color: transparent;
        }
        
        /* 滚动条 */
        QScrollBar:vertical {
            background-color: #F0F0F0;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #CCCCCC;
            min-height: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #999999;
        }
        
        QScrollBar::handle:vertical:pressed {
            background-color: #666666;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #F0F0F0;
            height: 12px;
            border: none;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #CCCCCC;
            min-width: 20px;
            border-radius: 6px;
            margin: 2px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #999999;
        }
        
        QScrollBar::handle:horizontal:pressed {
            background-color: #666666;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* 滑块 */
        QSlider::groove:horizontal {
            background-color: #CCCCCC;
            height: 4px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background-color: #2196F3;
            border: 1px solid #1976D2;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #1976D2;
        }
        
        QSlider::handle:horizontal:pressed {
            background-color: #1565C0;
        }
        
        /* 进度条 */
        QProgressBar {
            background-color: #FFFFFF;
            border: 1px solid #CCCCCC;
            border-radius: 2px;
            text-align: center;
            color: #212529;
            height: 20px;
        }
        
        QProgressBar::chunk {
            background-color: #2196F3;
            border-radius: 1px;
        }
        
        /* 列表和表格 */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            selection-background-color: #2196F3;
            selection-color: white;
            alternate-background-color: #F8F8F8;
        }
        
        /* 分割器 */
        QSplitter::handle {
            background-color: #CCCCCC;
        }
        
        QSplitter::handle:horizontal {
            width: 1px;
        }
        
        QSplitter::handle:vertical {
            height: 1px;
        }
        
        QSplitter::handle:hover {
            background-color: #999999;
        }
        
        /* 滚动区域 */
        QScrollArea {
            background-color: #F5F5F5;
            border: none;
        }
        
        /* 工具提示 */
        QToolTip {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            padding: 4px;
            border-radius: 2px;
        }
        
        /* 对话框 */
        QDialog {
            background-color: #F5F5F5;
            color: #212529;
        }
        
        QMessageBox {
            background-color: #F5F5F5;
            color: #212529;
        }
        
        QMessageBox QLabel {
            color: #212529;
        }
        
        /* 框架 */
        QFrame {
            background-color: transparent;
            color: #212529;
        }
        
        /* 旋钮框 */
        QSpinBox, QDoubleSpinBox {
            background-color: #FFFFFF;
            color: #212529;
            border: 1px solid #CCCCCC;
            border-radius: 2px;
            padding: 4px;
            min-height: 22px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #2196F3;
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            background-color: #F0F0F0;
            border-left: 1px solid #CCCCCC;
            border-top-right-radius: 2px;
        }
        
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
            background-color: #E0E0E0;
        }
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #F0F0F0;
            border-left: 1px solid #CCCCCC;
            border-bottom-right-radius: 2px;
        }
        
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #E0E0E0;
        }
        """


class GSSEGUI(QMainWindow):
    # 定义信号，用于从后台线程安全地更新UI
    log_signal = pyqtSignal(str, str)  # message, level
    status_signal = pyqtSignal(str, object)  # message, progress (可以是None)
    system_info_signal = pyqtSignal(str)  # info
    show_message_signal = pyqtSignal(str, str, str)  # title, message, msg_type (critical/warning/info)
    
    # UI更新信号（线程安全）
    set_text_signal = pyqtSignal(object, str)  # widget, text
    set_enabled_signal = pyqtSignal(object, bool)  # widget, enabled
    colmap_status_signal = pyqtSignal(str)  # status text
    question_dialog_signal = pyqtSignal(str, str, str)  # title, message, callback_name
    update_editor_view_signal = pyqtSignal()  # 更新编辑器视图
    display_view_signal = pyqtSignal()  # 显示当前视图
    
    def __init__(self):
        super().__init__()
        
        # 连接信号到槽函数
        self.log_signal.connect(self._log_internal)
        self.status_signal.connect(self._update_status_internal)
        self.system_info_signal.connect(self._update_system_info_internal)
        self.show_message_signal.connect(self._show_message_internal)
        self.set_text_signal.connect(self._set_text_internal)
        self.set_enabled_signal.connect(self._set_enabled_internal)
        self.colmap_status_signal.connect(self._set_colmap_status_internal)
        self.question_dialog_signal.connect(self._show_question_dialog)
        self.update_editor_view_signal.connect(self._update_editor_view_internal)
        self.display_view_signal.connect(self._display_view_internal)
        
        # 初始化内存管理
        MemoryManager.optimize_memory_settings()
        
        # 状态变量
        self.model_loaded = False
        self.gaussians = None
        self.scene = None
        self.cameras = None
        self.pipeline = None
        self.background = None
        self.predictor = None
        self.sam_features = {}
        self.current_view_idx = 0
        self.current_image = None
        self.current_mask = None
        self.click_points = []
        self.current_point_type = 1  # 1=正点, 0=负点
        self.use_cpu_fallback = False
        
        # 预处理相关状态变量
        self.preprocess_points = []
        self.preprocess_current_point_type = 1
        self.preprocess_predictor = None
        self.preprocess_sam_features = {}
        self.preprocess_mask_id = 2
        self.original_images = {}
        
        # 训练相关状态变量
        self.training_process = None
        self.training_thread = None
        self.is_training = False
        
        # SOG训练相关状态变量
        self.sog_process = None
        self.sog_thread = None
        self.is_sog_training = False
        self.sog_trainer = None
        
        # COLMAP相关状态变量
        self.colmap_processor = None
        self.colmap_thread = None
        self.is_colmap_processing = False
        
        # 3D渲染相关
        self.sibr_process = None
        self.temp_ply_path = None
        # 使用相对路径，基于项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.sibr_viewer_path = os.path.join(project_root, "dependencies", "SIBR_viewers", "install", "bin", "SIBR_gaussianViewer_app")
        
        # SAGA相关状态变量
        self.saga_module = None
        self.saga_feature_loaded = False
        self.saga_point_prompts = []
        self.saga_current_point_type = 1
        
        # 编辑相关状态变量
        self.gaussian_editor = None
        self.edit_mode = False
        self.current_edit_type = EditType.NONE if EDITOR_AVAILABLE else None
        self.current_view_mode = 'splat'  # 默认视图模式
        
        # 分割对比相关状态变量
        self.original_gaussians = None  # 保存原始高斯模型
        self.segmented_gaussians = None  # 保存分割后的高斯模型
        self.segmented_editor = None  # 保存分割后模型的编辑器
        self.is_showing_segmented = False  # 当前是否显示分割后的模型
        
        # GIS相关状态变量
        self.cesium_widget = None  # Cesium地图查看器
        self.gis_converter = None  # GIS数据转换器
        self.geo_config = None  # 地理数据配置
        self.coordinate_transformer = None  # 坐标转换器
        
        # HTTP服务器相关状态变量
        self.http_server = None  # HTTP服务器实例
        self.http_server_started = False  # HTTP服务器是否已启动
        
        # 主题管理
        self.settings = QSettings("GSSE", "GSSEGUI")
        self.current_theme = self.settings.value("theme", "dark", type=str)  # 默认深色主题
        self.app = QApplication.instance()
        
        # 初始化默认颜色常量（深色主题）
        self.status_color = "#5BA3D8"
        self.hint_color = "#B0B0B0"
        self.text_color = "#E0E0E0"
        self.bg_dark = "#2B2B2B"
        self.bg_panel = "#252525"
        self.border_color = "#404040"
        self.edit_title_bg = "#252525"
        self.edit_title_border = "#404040"
        self.group_bg = "#2B2B2B"
        self.info_bg = "#2B2B2B"
        self.info_border = "#404040"
        self.btn_primary = "#0E639C"
        self.btn_primary_hover = "#1177BB"
        self.btn_danger = "#C75050"
        self.btn_danger_hover = "#D96A6A"
        
        # 初始化UI
        self.init_ui()
        
        # 应用主题（这会更新颜色常量）
        self.apply_theme(self.current_theme)
        
        # 设置定时器更新系统信息
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_info)
        self.timer.start(5000)  # 每5秒更新一次
            
            # 设置定时器更新3D视图相机参数
        if EDITOR_AVAILABLE:
            self.viewer_timer = QTimer()
            self.viewer_timer.timeout.connect(self.update_viewer_camera_params)
            self.viewer_timer.start(100)  # 每100ms更新一次相机参数
        
        self.update_system_info()
        
        # 启动HTTP服务器
        self.start_http_server()
        
        # 添加欢迎日志
        self.log("=== GSSE (Gaussian Splatting Semantic Editor) 图形化界面已启动 ===", "info")
        self.log("请先加载3DGS模型，然后进行分割操作", "info")
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("GSSE - Gaussian Splatting Semantic Editor")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 0)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建水平分割器：左侧控制面板 + 右侧内容区
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # 左侧控制面板（可滚动）
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setMinimumWidth(400)
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        self.create_control_panel(left_layout)
        self.left_scroll.setWidget(self.left_widget)
        self.main_splitter.addWidget(self.left_scroll)
        
        # 右侧内容区：中间显示区域 + 右侧日志
        right_splitter = QSplitter(Qt.Horizontal)
        
        # 中间显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0, 0, 0, 0)
        self.create_display_panel(display_layout)
        right_splitter.addWidget(display_widget)
        
        # 右侧日志栏
        self.log_widget = QWidget()
        log_layout = QVBoxLayout(self.log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.create_log_panel(log_layout)
        self.log_widget.setMinimumWidth(350)
        right_splitter.addWidget(self.log_widget)
        
        self.main_splitter.addWidget(right_splitter)
        
        # 设置分割器比例
        right_splitter.setSizes([800, 400])
        
        # 计算合适的边栏宽度
        # 强制更新布局以获取正确的尺寸提示
        self.left_widget.updateGeometry()
        self.left_scroll.updateGeometry()
        
        # 获取控制面板的理想宽度（包括边距）
        # 添加一些额外的空间用于滚动条和边距（约20-30像素）
        ideal_width = self.left_widget.sizeHint().width()
        if ideal_width <= 0:
            ideal_width = self.left_widget.minimumSizeHint().width()
        
        # 设置合理的宽度范围：最小400，最大650
        sidebar_width = max(400, min(650, ideal_width + 30))
        self.left_scroll.setMinimumWidth(sidebar_width)
        
        # 设置分割器初始大小
        # 根据窗口初始大小调整，确保边栏能完全显示内容
        window_width = self.width()
        if window_width > 0:
            right_width = window_width - sidebar_width - 50  # 50像素用于分割器和其他边距
            if right_width < 600:
                right_width = 600
            self.main_splitter.setSizes([sidebar_width, right_width])
        else:
            # 如果窗口还未显示，使用默认值
            self.main_splitter.setSizes([sidebar_width, 1200])
        
        # 创建状态栏
        self.create_status_bar()
        
        # 设置窗口大小策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 标记是否需要调整边栏宽度（在窗口显示后）
        self._sidebar_adjusted = False
    
    def showEvent(self, event):
        """窗口显示事件，用于在首次显示后调整边栏宽度"""
        super().showEvent(event)
        
        # 仅在首次显示时调整一次
        if not self._sidebar_adjusted:
            self._sidebar_adjusted = True
            # 使用定时器延迟调整，确保布局已完成
            QTimer.singleShot(100, self._adjust_sidebar_width)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            # 停止HTTP服务器
            self.stop_http_server()
            
            # 停止所有正在运行的进程
            if self.is_colmap_processing and self.colmap_processor:
                self.colmap_processor.stop_processing()
            
            if self.is_training and self.training_process:
                self.training_process.terminate()
                
            if self.is_sog_training and self.sog_process:
                self.sog_process.terminate()
                
            if self.sibr_process:
                self.sibr_process.terminate()
            
            # 清理GPU内存
            MemoryManager.clear_gpu_memory()
            
            self.log("程序正在关闭，已清理所有资源", "info")
            
        except Exception as e:
            print(f"关闭程序时出错: {e}")
        
        # 接受关闭事件
        event.accept()
    
    def _adjust_sidebar_width(self):
        """调整边栏宽度以适应内容"""
        if not hasattr(self, 'left_widget') or not hasattr(self, 'main_splitter'):
            return
        
        try:
            # 强制更新布局
            self.left_widget.updateGeometry()
            self.left_widget.adjustSize()
            
            # 获取控制面板的理想宽度
            ideal_width = self.left_widget.sizeHint().width()
            if ideal_width <= 0:
                ideal_width = self.left_widget.minimumSizeHint().width()
            
            # 如果还是无效，使用实际布局后的宽度
            if ideal_width <= 0:
                ideal_width = self.left_widget.width()
            
            # 添加额外的空间用于滚动条和边距
            # 设置合理的宽度范围：最小400，最大650
            sidebar_width = max(400, min(650, ideal_width + 30))
            
            # 更新分割器大小
            current_sizes = self.main_splitter.sizes()
            if len(current_sizes) >= 2:
                window_width = self.width()
                if window_width > 0:
                    right_width = window_width - sidebar_width - 50  # 50像素用于分割器和其他边距
                    if right_width < 600:
                        right_width = 600
                    self.main_splitter.setSizes([sidebar_width, right_width])
                else:
                    self.main_splitter.setSizes([sidebar_width, current_sizes[1]])
        except Exception as e:
            # 如果调整失败，使用默认值
            print(f"[DEBUG] 调整边栏宽度时出错: {e}", file=sys.stderr)

    def toggle_left_panel(self):
        """切换左侧控制面板可见性"""
        try:
            if hasattr(self, 'left_scroll') and self.left_scroll is not None:
                self.left_scroll.setVisible(not self.left_scroll.isVisible())
        except Exception as e:
            self.log(f"切换左侧面板失败: {e}", "warning")

    def toggle_log_panel(self):
        """切换右侧日志面板可见性"""
        try:
            if hasattr(self, 'log_widget') and self.log_widget is not None:
                self.log_widget.setVisible(not self.log_widget.isVisible())
        except Exception as e:
            self.log(f"切换日志面板失败: {e}", "warning")

    def open_workspace_dir(self):
        """打开程序工作目录"""
        try:
            base_dir = os.path.dirname(__file__)
            QDesktopServices.openUrl(QUrl.fromLocalFile(base_dir))
        except Exception as e:
            self.show_message("错误", f"无法打开目录: {e}", "error")

    def action_load_images(self):
        """通过对话框选择图像目录并加载原始图像"""
        try:
            directory = QFileDialog.getExistingDirectory(self, "选择图像目录")
            if directory:
                self.load_original_images(directory)
        except Exception as e:
            self.show_message("错误", f"加载原始图像失败: {e}", "error")
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        file_menu.addAction('加载模型(&O)', self.load_model, QKeySequence('Ctrl+O'))
        file_menu.addAction('加载原始图像目录(&I)', self.action_load_images)
        file_menu.addSeparator()
        file_menu.addAction('保存分割结果(&S)', self.save_results, QKeySequence('Ctrl+S'))
        file_menu.addAction('保存日志(&L)', self.save_log)
        file_menu.addSeparator()
        file_menu.addAction('生成Mesh(&M)', self.generate_mesh, QKeySequence('Ctrl+M'))
        file_menu.addAction('导出Mesh(&E)', self.export_mesh)
        file_menu.addAction('清除Mesh缓存', self.clear_mesh_cache)
        file_menu.addSeparator()
        file_menu.addAction('打开工作目录', self.open_workspace_dir)
        file_menu.addAction('重置系统(&R)', self.reset_system, QKeySequence('Ctrl+R'))
        file_menu.addSeparator()
        file_menu.addAction('退出(&X)', self.close, QKeySequence.Quit)

        # 运行菜单
        run_menu = menubar.addMenu('运行(&R)')
        run_menu.addAction('开始COLMAP处理', self.start_colmap_processing)
        run_menu.addAction('停止COLMAP处理', self.stop_colmap_processing)
        run_menu.addSeparator()
        run_menu.addAction('分割预处理', self.run_preprocess_segmentation)
        run_menu.addAction('运行分割 (F5)', self.run_segmentation, QKeySequence('F5'))
        run_menu.addSeparator()
        run_menu.addAction('一键全流程 (3DGS→分割)', self.start_full_workflow)
        run_menu.addAction('一键完整流程', self.start_full_workflow_with_sags)

        # 训练菜单
        train_menu = menubar.addMenu('训练(&T)')
        train_menu.addAction('开始3DGS训练', self.start_training, QKeySequence('Ctrl+T'))
        train_menu.addAction('停止3DGS训练', self.stop_training, QKeySequence('Ctrl+Shift+T'))
        train_menu.addSeparator()
        train_menu.addAction('开始SOG训练', self.start_sog_training)
        train_menu.addAction('停止SOG训练', self.stop_sog_training)

        # 视图菜单
        view_menu = menubar.addMenu('视图(&V)')
        view_menu.addAction('显示3D模型', self.show_3d_model, QKeySequence('F7'))
        view_menu.addAction('停止3D查看器', self.stop_3d_viewer)
        view_menu.addSeparator()
        view_menu.addAction('切换左侧控制面板', self.toggle_left_panel)
        view_menu.addAction('切换右侧日志面板', self.toggle_log_panel)
        view_menu.addSeparator()
        # 主题切换
        theme_menu = view_menu.addMenu('主题(&T)')
        theme_menu.addAction('深色主题', lambda: self.apply_theme('dark'))
        theme_menu.addAction('浅色主题', lambda: self.apply_theme('light'))

        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        help_menu.addAction('快捷键(&K)', self.show_shortcuts, QKeySequence('F1'))
        help_menu.addAction('关于(&A)', self.show_about)
    
    def create_control_panel(self, parent_layout):
        """创建左侧控制面板"""
        # 创建主标签页容器
        control_tabs = QTabWidget()
        
        # ========== 标签页1: 模型训练 ==========
        training_tab = QWidget()
        training_tab_layout = QVBoxLayout(training_tab)
        training_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # COLMAP处理区域
        colmap_group = QGroupBox("COLMAP处理")
        colmap_layout = QVBoxLayout()
        
        # 输入类型选择
        input_type_layout = QHBoxLayout()
        input_type_layout.addWidget(QLabel("输入类型:"))
        self.colmap_input_type_group = QButtonGroup()
        self.colmap_images_radio = QRadioButton("图像目录")
        self.colmap_images_radio.setChecked(True)
        self.colmap_video_radio = QRadioButton("视频文件")
        self.colmap_input_type_group.addButton(self.colmap_images_radio, 0)
        self.colmap_input_type_group.addButton(self.colmap_video_radio, 1)
        self.colmap_images_radio.toggled.connect(self.on_colmap_input_type_change)
        self.colmap_video_radio.toggled.connect(self.on_colmap_input_type_change)
        input_type_layout.addWidget(self.colmap_images_radio)
        input_type_layout.addWidget(self.colmap_video_radio)
        input_type_layout.addStretch()
        colmap_layout.addLayout(input_type_layout)
        
        # COLMAP输入路径
        colmap_layout.addWidget(QLabel("输入路径:"))
        input_path_layout = QHBoxLayout()
        sags_dir = os.path.dirname(__file__)
        default_input_path = os.path.join(sags_dir, "colmap", "input")
        self.colmap_input_path_edit = QLineEdit(default_input_path)
        input_path_btn = QPushButton("浏览")
        input_path_btn.clicked.connect(self.browse_colmap_input_path)
        input_path_layout.addWidget(self.colmap_input_path_edit)
        input_path_layout.addWidget(input_path_btn)
        colmap_layout.addLayout(input_path_layout)
        
        # 视频处理参数（默认隐藏）
        self.video_params_group = QGroupBox("视频处理参数")
        video_params_layout = QVBoxLayout()
        
        video_params_inner = QHBoxLayout()
        video_left = QVBoxLayout()
        video_left.addWidget(QLabel("提取帧率 (帧/秒):"))
        self.video_frame_rate_edit = QLineEdit("1")
        video_left.addWidget(self.video_frame_rate_edit)
        video_left.addWidget(QLabel("图像质量 (1-100):"))
        self.video_quality_edit = QLineEdit("95")
        video_left.addWidget(self.video_quality_edit)
        
        video_right = QVBoxLayout()
        video_right.addWidget(QLabel("最大帧数 (0=不限制):"))
        self.video_max_frames_edit = QLineEdit("0")
        video_right.addWidget(self.video_max_frames_edit)
        video_right.addWidget(QLabel("调整宽度 (0=原始):"))
        self.video_resize_width_edit = QLineEdit("0")
        video_right.addWidget(self.video_resize_width_edit)
        
        video_params_inner.addLayout(video_left)
        video_params_inner.addLayout(video_right)
        video_params_layout.addLayout(video_params_inner)
        
        self.video_info_label = QLabel("")
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet("color: #5BA3D8;")
        video_params_layout.addWidget(self.video_info_label)
        
        self.video_params_group.setLayout(video_params_layout)
        colmap_layout.addWidget(self.video_params_group)
        # 默认隐藏视频参数
        self.video_params_group.setVisible(False)
        
        # 自动路径管理说明
        path_info = QLabel("输出路径将自动设置为: gaussiansplatting/input/输入文件夹名")
        path_info.setWordWrap(True)
        path_info.setStyleSheet("color: #B0B0B0;")
        colmap_layout.addWidget(path_info)
        
        # COLMAP参数
        colmap_params_layout = QHBoxLayout()
        colmap_left = QVBoxLayout()
        colmap_left.addWidget(QLabel("相机模型:"))
        self.colmap_camera_model_combo = QComboBox()
        self.colmap_camera_model_combo.addItems(["OPENCV", "PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"])
        self.colmap_camera_model_combo.setCurrentText("SIMPLE_PINHOLE")
        colmap_left.addWidget(self.colmap_camera_model_combo)
        
        colmap_left.addWidget(QLabel("处理质量:"))
        self.colmap_quality_combo = QComboBox()
        self.colmap_quality_combo.addItems(["Low", "Medium", "High", "Extreme"])
        self.colmap_quality_combo.setCurrentText("Medium")
        colmap_left.addWidget(self.colmap_quality_combo)
        
        colmap_left.addWidget(QLabel("数据类型:"))
        self.colmap_data_type_combo = QComboBox()
        self.colmap_data_type_combo.addItems(["Individual images", "Video"])
        self.colmap_data_type_combo.setCurrentText("Individual images")
        colmap_left.addWidget(self.colmap_data_type_combo)
        
        colmap_left.addWidget(QLabel("映射器类型:"))
        self.colmap_mapper_type_combo = QComboBox()
        self.colmap_mapper_type_combo.addItems(["incremental", "global"])
        self.colmap_mapper_type_combo.setCurrentText("incremental")
        colmap_left.addWidget(self.colmap_mapper_type_combo)
        
        colmap_right = QVBoxLayout()
        colmap_right.addWidget(QLabel("线程数 (-1=自动):"))
        self.colmap_num_threads_edit = QLineEdit("-1")
        colmap_right.addWidget(self.colmap_num_threads_edit)
        
        self.colmap_single_camera_check = QCheckBox("单相机模式")
        self.colmap_single_camera_check.setChecked(True)
        colmap_right.addWidget(self.colmap_single_camera_check)
        
        self.colmap_sparse_model_check = QCheckBox("稀疏模型")
        self.colmap_sparse_model_check.setChecked(True)
        colmap_right.addWidget(self.colmap_sparse_model_check)
        
        self.colmap_dense_model_check = QCheckBox("密集模型")
        self.colmap_dense_model_check.setChecked(True)
        colmap_right.addWidget(self.colmap_dense_model_check)
        
        self.colmap_use_gpu_check = QCheckBox("GPU加速")
        self.colmap_use_gpu_check.setChecked(True)
        colmap_right.addWidget(self.colmap_use_gpu_check)
        
        colmap_params_layout.addLayout(colmap_left)
        colmap_params_layout.addLayout(colmap_right)
        colmap_layout.addLayout(colmap_params_layout)
        
        # 词汇树文件
        colmap_layout.addWidget(QLabel("词汇树文件 (可选):"))
        vocab_layout = QHBoxLayout()
        default_vocab_path = os.path.join(os.path.dirname(__file__), "colmap", "vocabulary_tree", "vocab_tree_flickr100K_words32K.bin")
        self.colmap_vocab_tree_edit = QLineEdit(default_vocab_path)
        vocab_btn = QPushButton("浏览")
        vocab_btn.clicked.connect(self.browse_vocab_tree)
        vocab_layout.addWidget(self.colmap_vocab_tree_edit)
        vocab_layout.addWidget(vocab_btn)
        colmap_layout.addLayout(vocab_layout)
        
        # COLMAP控制按钮
        colmap_control_layout = QHBoxLayout()
        self.start_colmap_btn = QPushButton("开始COLMAP处理")
        self.start_colmap_btn.clicked.connect(self.start_colmap_processing)
        self.stop_colmap_btn = QPushButton("停止COLMAP处理")
        self.stop_colmap_btn.clicked.connect(self.stop_colmap_processing)
        self.stop_colmap_btn.setEnabled(False)
        colmap_control_layout.addWidget(self.start_colmap_btn)
        colmap_control_layout.addWidget(self.stop_colmap_btn)
        colmap_layout.addLayout(colmap_control_layout)
        
        # COLMAP状态显示
        self.colmap_status_label = QLabel("未开始COLMAP处理")
        self.colmap_status_label.setStyleSheet("color: #5BA3D8;")
        colmap_layout.addWidget(self.colmap_status_label)
        
        colmap_group.setLayout(colmap_layout)
        training_tab_layout.addWidget(colmap_group)
        
        # 训练区域
        training_group = QGroupBox("3DGS训练")
        training_layout = QVBoxLayout()
        
        training_layout.addWidget(QLabel("训练数据路径:"))
        training_data_layout = QHBoxLayout()
        sags_dir = os.path.dirname(__file__)
        default_input_path = os.path.join(sags_dir, "gaussiansplatting", "input")
        self.training_data_path_edit = QLineEdit(default_input_path)
        training_data_btn = QPushButton("浏览")
        training_data_btn.clicked.connect(self.browse_training_data_path)
        training_data_layout.addWidget(self.training_data_path_edit)
        training_data_layout.addWidget(training_data_btn)
        training_layout.addLayout(training_data_layout)
        
        training_layout.addWidget(QLabel("输出路径:"))
        training_output_layout = QHBoxLayout()
        default_output_path = os.path.join(sags_dir, "gaussiansplatting", "output")
        self.training_output_path_edit = QLineEdit(default_output_path)
        training_output_btn = QPushButton("浏览")
        training_output_btn.clicked.connect(self.browse_training_output_path)
        training_output_layout.addWidget(self.training_output_path_edit)
        training_output_layout.addWidget(training_output_btn)
        training_layout.addLayout(training_output_layout)
        
        # 训练参数
        training_params_layout = QHBoxLayout()
        training_left = QVBoxLayout()
        training_left.addWidget(QLabel("迭代次数:"))
        self.training_iterations_edit = QLineEdit("30000")
        training_left.addWidget(self.training_iterations_edit)
        training_left.addWidget(QLabel("分辨率:"))
        self.training_resolution_combo = QComboBox()
        self.training_resolution_combo.addItems(["1", "2", "4", "8", "-1"])
        self.training_resolution_combo.setCurrentText("4")
        training_left.addWidget(self.training_resolution_combo)
        training_left.addWidget(QLabel("学习率:"))
        self.training_lr_edit = QLineEdit("0.00016")
        training_left.addWidget(self.training_lr_edit)
        
        training_right = QVBoxLayout()
        training_right.addWidget(QLabel("密化开始:"))
        self.densify_from_edit = QLineEdit("500")
        training_right.addWidget(self.densify_from_edit)
        training_right.addWidget(QLabel("密化结束:"))
        self.densify_until_edit = QLineEdit("15000")
        training_right.addWidget(self.densify_until_edit)
        training_right.addWidget(QLabel("测试间隔:"))
        self.test_interval_edit = QLineEdit("7000")
        training_right.addWidget(self.test_interval_edit)
        
        training_params_layout.addLayout(training_left)
        training_params_layout.addLayout(training_right)
        training_layout.addLayout(training_params_layout)
        
        # 训练控制按钮
        training_control_layout = QHBoxLayout()
        self.start_training_btn = QPushButton("开始训练")
        self.start_training_btn.clicked.connect(self.start_training)
        self.stop_training_btn = QPushButton("停止训练")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        training_control_layout.addWidget(self.start_training_btn)
        training_control_layout.addWidget(self.stop_training_btn)
        training_layout.addLayout(training_control_layout)
        
        self.training_status_label = QLabel("未开始训练")
        self.training_status_label.setStyleSheet("color: #5BA3D8;")
        training_layout.addWidget(self.training_status_label)
        
        training_group.setLayout(training_layout)
        training_tab_layout.addWidget(training_group)
        
        # SOG训练区域
        sog_training_group = QGroupBox("SOG训练")
        sog_training_layout = QVBoxLayout()
        
        sog_info_label = QLabel("SOG训练可以得到更小的模型文件，同时保持较高质量")
        sog_info_label.setWordWrap(True)
        sog_info_label.setStyleSheet("color: #B0B0B0;")
        sog_training_layout.addWidget(sog_info_label)
        
        sog_training_layout.addWidget(QLabel("训练数据路径:"))
        sog_data_layout = QHBoxLayout()
        default_sog_input_path = os.path.join(sags_dir, "gaussiansplatting", "input")
        self.sog_data_path_edit = QLineEdit(default_sog_input_path)
        sog_data_btn = QPushButton("浏览")
        sog_data_btn.clicked.connect(self.browse_sog_data_path)
        sog_data_layout.addWidget(self.sog_data_path_edit)
        sog_data_layout.addWidget(sog_data_btn)
        sog_training_layout.addLayout(sog_data_layout)
        
        sog_training_layout.addWidget(QLabel("输出路径:"))
        sog_output_layout = QHBoxLayout()
        default_sog_output_path = os.path.join(sags_dir, "gaussiansplatting", "output", "sog_output")
        self.sog_output_path_edit = QLineEdit(default_sog_output_path)
        sog_output_btn = QPushButton("浏览")
        sog_output_btn.clicked.connect(self.browse_sog_output_path)
        sog_output_layout.addWidget(self.sog_output_path_edit)
        sog_output_layout.addWidget(sog_output_btn)
        sog_training_layout.addLayout(sog_output_layout)
        
        # SOG训练参数
        sog_params_layout = QHBoxLayout()
        sog_left = QVBoxLayout()
        sog_left.addWidget(QLabel("迭代次数:"))
        self.sog_iterations_edit = QLineEdit("30000")
        sog_left.addWidget(self.sog_iterations_edit)
        sog_left.addWidget(QLabel("配置文件:"))
        self.sog_config_combo = QComboBox()
        self.sog_config_combo.addItems(["ours_q_sh_local_test", "custom"])
        self.sog_config_combo.setCurrentText("ours_q_sh_local_test")
        sog_left.addWidget(self.sog_config_combo)
        
        sog_right = QVBoxLayout()
        self.sog_use_sh_check = QCheckBox("使用球谐函数")
        self.sog_use_sh_check.setChecked(True)
        sog_right.addWidget(self.sog_use_sh_check)
        sog_right.addWidget(QLabel("压缩迭代:"))
        self.sog_compress_iter_edit = QLineEdit("7000,10000,20000,30000")
        sog_right.addWidget(self.sog_compress_iter_edit)
        
        sog_params_layout.addLayout(sog_left)
        sog_params_layout.addLayout(sog_right)
        sog_training_layout.addLayout(sog_params_layout)
        
        # SOG训练控制按钮
        sog_control_layout = QHBoxLayout()
        self.start_sog_training_btn = QPushButton("开始SOG训练")
        self.start_sog_training_btn.clicked.connect(self.start_sog_training)
        self.stop_sog_training_btn = QPushButton("停止SOG训练")
        self.stop_sog_training_btn.clicked.connect(self.stop_sog_training)
        self.stop_sog_training_btn.setEnabled(False)
        sog_control_layout.addWidget(self.start_sog_training_btn)
        sog_control_layout.addWidget(self.stop_sog_training_btn)
        sog_training_layout.addLayout(sog_control_layout)
        
        self.sog_training_status_label = QLabel("未开始SOG训练")
        self.sog_training_status_label.setStyleSheet("color: #5BA3D8;")
        sog_training_layout.addWidget(self.sog_training_status_label)
        
        sog_training_group.setLayout(sog_training_layout)
        training_tab_layout.addWidget(sog_training_group)
        
        # 一键式工作流按钮（放在训练标签页最后）
        workflow_group = QGroupBox("一键式工作流")
        workflow_layout = QVBoxLayout()
        
        self.full_workflow_btn = QPushButton("一键式Colmap+3DGS处理")
        self.full_workflow_btn.clicked.connect(self.start_full_workflow)
        workflow_layout.addWidget(self.full_workflow_btn)
        
        workflow_group.setLayout(workflow_layout)
        training_tab_layout.addWidget(workflow_group)
        
        training_tab_layout.addStretch()
        control_tabs.addTab(training_tab, "模型训练")
        
        # ========== 标签页2: SAGS分割 ==========
        sags_tab = QWidget()
        sags_tab_layout = QVBoxLayout(sags_tab)
        sags_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # 模型加载区域
        model_group = QGroupBox("模型加载")
        model_layout = QVBoxLayout()
        
        model_layout.addWidget(QLabel("模型路径:"))
        model_path_layout = QHBoxLayout()
        default_model_path = os.path.join(sags_dir, "gaussiansplatting", "output")
        self.model_path_edit = QLineEdit(default_model_path)
        model_path_btn = QPushButton("浏览")
        model_path_btn.clicked.connect(self.browse_model_path)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(model_path_btn)
        model_layout.addLayout(model_path_layout)
        
        model_layout.addWidget(QLabel("数据路径:"))
        source_path_layout = QHBoxLayout()
        default_source_path = os.path.join(sags_dir, "gaussiansplatting", "input")
        self.source_path_edit = QLineEdit(default_source_path)
        source_path_btn = QPushButton("浏览")
        source_path_btn.clicked.connect(self.browse_source_path)
        source_path_layout.addWidget(self.source_path_edit)
        source_path_layout.addWidget(source_path_btn)
        model_layout.addLayout(source_path_layout)
        
        model_layout.addWidget(QLabel("迭代次数:"))
        self.iteration_edit = QLineEdit("7000")
        model_layout.addWidget(self.iteration_edit)
        
        model_layout.addWidget(QLabel("分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1", "2", "4", "8", "-1"])
        self.resolution_combo.setCurrentText("4")
        model_layout.addWidget(self.resolution_combo)
        
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        model_group.setLayout(model_layout)
        sags_tab_layout.addWidget(model_group)
        
        # 分割参数区域
        seg_group = QGroupBox("分割参数")
        seg_layout = QVBoxLayout()
        
        seg_layout.addWidget(QLabel("投票阈值:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(lambda v: self.threshold_label.setText(f"当前值: {v/100:.2f}"))
        seg_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("当前值: 0.70")
        seg_layout.addWidget(self.threshold_label)
        
        seg_layout.addWidget(QLabel("SAM模型:"))
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.addItems(["vit_h", "vit_l", "vit_b", "fastsam_s", "fastsam_x"])
        self.sam_model_combo.setCurrentText("vit_b")
        seg_layout.addWidget(self.sam_model_combo)
        
        seg_layout.addWidget(QLabel("SAM长边:"))
        self.sam_long_side_combo = QComboBox()
        self.sam_long_side_combo.addItems(["384", "512", "640", "768", "1024"])
        self.sam_long_side_combo.setCurrentText("640")
        seg_layout.addWidget(self.sam_long_side_combo)
        
        # FastSAM参数控制
        seg_layout.addWidget(QLabel("FastSAM置信度阈值:"))
        self.fastsam_conf_slider = QSlider(Qt.Horizontal)
        self.fastsam_conf_slider.setMinimum(10)
        self.fastsam_conf_slider.setMaximum(100)
        self.fastsam_conf_slider.setValue(50)
        self.fastsam_conf_slider.valueChanged.connect(lambda v: self.fastsam_conf_label.setText(f"当前值: {v/100:.2f} (推荐0.5-0.6)"))
        seg_layout.addWidget(self.fastsam_conf_slider)
        self.fastsam_conf_label = QLabel("当前值: 0.50 (推荐0.5-0.6)")
        seg_layout.addWidget(self.fastsam_conf_label)
        
        seg_layout.addWidget(QLabel("FastSAM IoU阈值:"))
        self.fastsam_iou_slider = QSlider(Qt.Horizontal)
        self.fastsam_iou_slider.setMinimum(30)
        self.fastsam_iou_slider.setMaximum(90)
        self.fastsam_iou_slider.setValue(70)
        self.fastsam_iou_slider.valueChanged.connect(lambda v: self.fastsam_iou_label.setText(f"当前值: {v/100:.2f} (推荐0.6-0.8)"))
        seg_layout.addWidget(self.fastsam_iou_slider)
        self.fastsam_iou_label = QLabel("当前值: 0.70 (推荐0.6-0.8)")
        seg_layout.addWidget(self.fastsam_iou_label)
        
        seg_layout.addWidget(QLabel("GD间隔:"))
        self.gd_interval_combo = QComboBox()
        self.gd_interval_combo.addItems(["-1", "1", "2", "4", "8"])
        self.gd_interval_combo.setCurrentText("4")
        seg_layout.addWidget(self.gd_interval_combo)
        
        # Mask后处理选项
        self.enable_postprocess_check = QCheckBox("启用后处理（形态学操作和边界平滑）")
        self.enable_postprocess_check.setChecked(True)
        seg_layout.addWidget(self.enable_postprocess_check)
        
        seg_group.setLayout(seg_layout)
        sags_tab_layout.addWidget(seg_group)
        
        # 提示方式区域
        prompt_group = QGroupBox("提示方式")
        prompt_layout = QVBoxLayout()
        
        self.prompt_type_group = QButtonGroup()
        self.point_prompt_radio = QRadioButton("点提示")
        self.point_prompt_radio.setChecked(True)
        self.text_prompt_radio = QRadioButton("文本提示")
        self.prompt_type_group.addButton(self.point_prompt_radio, 0)
        self.prompt_type_group.addButton(self.text_prompt_radio, 1)
        self.point_prompt_radio.toggled.connect(self.on_prompt_type_change)
        prompt_layout.addWidget(self.point_prompt_radio)
        prompt_layout.addWidget(self.text_prompt_radio)
        
        # 点类型选择
        point_type_layout = QHBoxLayout()
        point_type_layout.addWidget(QLabel("点类型:"))
        self.point_type_group = QButtonGroup()
        self.positive_point_radio = QRadioButton("正点")
        self.positive_point_radio.setChecked(True)
        self.negative_point_radio = QRadioButton("负点")
        self.point_type_group.addButton(self.positive_point_radio, 1)
        self.point_type_group.addButton(self.negative_point_radio, 0)
        self.positive_point_radio.toggled.connect(lambda: setattr(self, 'current_point_type', 1))
        self.negative_point_radio.toggled.connect(lambda: setattr(self, 'current_point_type', 0))
        point_type_layout.addWidget(self.positive_point_radio)
        point_type_layout.addWidget(self.negative_point_radio)
        point_type_layout.addStretch()
        prompt_layout.addLayout(point_type_layout)
        
        # 点提示管理按钮
        point_manage_layout = QHBoxLayout()
        clear_points_btn = QPushButton("清除所有点")
        clear_points_btn.clicked.connect(self.clear_all_points)
        remove_point_btn = QPushButton("撤销最后一点")
        remove_point_btn.clicked.connect(self.remove_last_point)
        point_manage_layout.addWidget(clear_points_btn)
        point_manage_layout.addWidget(remove_point_btn)
        prompt_layout.addLayout(point_manage_layout)
        
        # 点提示列表显示
        prompt_layout.addWidget(QLabel("已添加的点:"))
        self.points_listbox = QListWidget()
        self.points_listbox.setMaximumHeight(60)
        prompt_layout.addWidget(self.points_listbox)
        
        prompt_layout.addWidget(QLabel("文本提示:"))
        self.text_prompt_edit = QLineEdit("chair")
        prompt_layout.addWidget(self.text_prompt_edit)
        
        prompt_layout.addWidget(QLabel("掩码ID (0/1/2):"))
        self.mask_id_combo = QComboBox()
        self.mask_id_combo.addItems(["0", "1", "2"])
        self.mask_id_combo.setCurrentText("2")
        prompt_layout.addWidget(self.mask_id_combo)
        
        # 调试模式
        self.debug_mode_check = QCheckBox("调试模式")
        self.debug_mode_check.setChecked(False)
        prompt_layout.addWidget(self.debug_mode_check)
        
        prompt_group.setLayout(prompt_layout)
        sags_tab_layout.addWidget(prompt_group)
        
        # 分割预处理区域
        preprocess_group = QGroupBox("分割预处理")
        preprocess_layout = QVBoxLayout()
        
        self.enable_preprocess_check = QCheckBox("启用分割预处理")
        self.enable_preprocess_check.setChecked(False)
        self.enable_preprocess_check.toggled.connect(self.on_preprocess_toggle)
        preprocess_layout.addWidget(self.enable_preprocess_check)
        
        # 加载原始图像按钮
        load_images_btn = QPushButton("加载原始图像目录")
        load_images_btn.clicked.connect(self.browse_original_images)
        preprocess_layout.addWidget(load_images_btn)
        
        # 预处理提示方式区域（默认隐藏）
        self.preprocess_prompt_group = QGroupBox("预处理提示方式")
        preprocess_prompt_layout = QVBoxLayout()
        
        self.preprocess_prompt_type_group = QButtonGroup()
        self.preprocess_point_prompt_radio = QRadioButton("点提示")
        self.preprocess_point_prompt_radio.setChecked(True)
        self.preprocess_text_prompt_radio = QRadioButton("文本提示")
        self.preprocess_prompt_type_group.addButton(self.preprocess_point_prompt_radio, 0)
        self.preprocess_prompt_type_group.addButton(self.preprocess_text_prompt_radio, 1)
        self.preprocess_point_prompt_radio.toggled.connect(self.on_preprocess_prompt_type_change)
        preprocess_prompt_layout.addWidget(self.preprocess_point_prompt_radio)
        preprocess_prompt_layout.addWidget(self.preprocess_text_prompt_radio)
        
        # 预处理点提示管理区域
        preprocess_point_layout = QVBoxLayout()
        
        preprocess_point_type_layout = QHBoxLayout()
        preprocess_point_type_layout.addWidget(QLabel("点类型:"))
        self.preprocess_point_type_group = QButtonGroup()
        self.preprocess_positive_point_radio = QRadioButton("正点")
        self.preprocess_positive_point_radio.setChecked(True)
        self.preprocess_negative_point_radio = QRadioButton("负点")
        self.preprocess_point_type_group.addButton(self.preprocess_positive_point_radio, 1)
        self.preprocess_point_type_group.addButton(self.preprocess_negative_point_radio, 0)
        self.preprocess_positive_point_radio.toggled.connect(lambda: setattr(self, 'preprocess_current_point_type', 1))
        self.preprocess_negative_point_radio.toggled.connect(lambda: setattr(self, 'preprocess_current_point_type', 0))
        preprocess_point_type_layout.addWidget(self.preprocess_positive_point_radio)
        preprocess_point_type_layout.addWidget(self.preprocess_negative_point_radio)
        preprocess_point_type_layout.addStretch()
        preprocess_point_layout.addLayout(preprocess_point_type_layout)
        
        preprocess_point_manage_layout = QHBoxLayout()
        clear_preprocess_points_btn = QPushButton("清除所有点")
        clear_preprocess_points_btn.clicked.connect(self.clear_all_preprocess_points)
        remove_preprocess_point_btn = QPushButton("撤销最后一点")
        remove_preprocess_point_btn.clicked.connect(self.remove_last_preprocess_point)
        preprocess_point_manage_layout.addWidget(clear_preprocess_points_btn)
        preprocess_point_manage_layout.addWidget(remove_preprocess_point_btn)
        preprocess_point_layout.addLayout(preprocess_point_manage_layout)
        
        preprocess_point_layout.addWidget(QLabel("已添加的点:"))
        self.preprocess_points_listbox = QListWidget()
        self.preprocess_points_listbox.setMaximumHeight(60)
        preprocess_point_layout.addWidget(self.preprocess_points_listbox)
        
        preprocess_prompt_layout.addLayout(preprocess_point_layout)
        
        # 预处理文本提示区域
        preprocess_text_layout = QVBoxLayout()
        preprocess_text_layout.addWidget(QLabel("文本提示:"))
        self.preprocess_text_edit = QLineEdit("")
        preprocess_text_layout.addWidget(self.preprocess_text_edit)
        preprocess_prompt_layout.addLayout(preprocess_text_layout)
        
        # 预处理参数
        preprocess_params_layout = QVBoxLayout()
        preprocess_params_layout.addWidget(QLabel("预处理SAM模型:"))
        self.preprocess_sam_model_combo = QComboBox()
        self.preprocess_sam_model_combo.addItems(["vit_h", "vit_l", "vit_b", "fastsam_s", "fastsam_x"])
        self.preprocess_sam_model_combo.setCurrentText("vit_b")
        preprocess_params_layout.addWidget(self.preprocess_sam_model_combo)
        
        preprocess_params_layout.addWidget(QLabel("预处理SAM长边:"))
        self.preprocess_sam_long_side_combo = QComboBox()
        self.preprocess_sam_long_side_combo.addItems(["384", "512", "640", "768", "1024"])
        self.preprocess_sam_long_side_combo.setCurrentText("640")
        preprocess_params_layout.addWidget(self.preprocess_sam_long_side_combo)
        
        preprocess_prompt_layout.addLayout(preprocess_params_layout)
        
        self.preprocess_prompt_group.setLayout(preprocess_prompt_layout)
        preprocess_layout.addWidget(self.preprocess_prompt_group)
        self.preprocess_prompt_group.setVisible(False)  # 默认隐藏
        
        preprocess_group.setLayout(preprocess_layout)
        sags_tab_layout.addWidget(preprocess_group)
        
        # 操作按钮区域
        action_group = QGroupBox("SAGS分割操作")
        action_layout = QVBoxLayout()
        
        preprocess_sam_btn = QPushButton("预处理SAM特征")
        preprocess_sam_btn.clicked.connect(self.preprocess_sam)
        action_layout.addWidget(preprocess_sam_btn)
        
        run_seg_btn = QPushButton("执行分割")
        run_seg_btn.clicked.connect(self.run_segmentation)
        action_layout.addWidget(run_seg_btn)
        
        save_results_btn = QPushButton("保存结果")
        save_results_btn.clicked.connect(self.save_results)
        action_layout.addWidget(save_results_btn)
        
        action_group.setLayout(action_layout)
        sags_tab_layout.addWidget(action_group)
        
        # 视图控制区域
        view_group = QGroupBox("视图控制")
        view_layout = QVBoxLayout()
        
        view_layout.addWidget(QLabel("当前视图:"))
        self.view_spinbox = QSpinBox()
        self.view_spinbox.setMinimum(0)
        self.view_spinbox.setMaximum(0)
        self.view_spinbox.valueChanged.connect(self.change_view)
        view_layout.addWidget(self.view_spinbox)
        
        view_buttons_layout = QHBoxLayout()
        prev_view_btn = QPushButton("上一视图")
        prev_view_btn.clicked.connect(self.prev_view)
        next_view_btn = QPushButton("下一视图")
        next_view_btn.clicked.connect(self.next_view)
        view_buttons_layout.addWidget(prev_view_btn)
        view_buttons_layout.addWidget(next_view_btn)
        view_layout.addLayout(view_buttons_layout)
        
        view_group.setLayout(view_layout)
        sags_tab_layout.addWidget(view_group)
        
        # 一键式工作流按钮（放在SAGS标签页最后）
        sags_workflow_group = QGroupBox("一键式工作流")
        sags_workflow_layout = QVBoxLayout()
        
        self.full_workflow_sags_btn = QPushButton("一键式Colmap+3DGS+分割处理")
        self.full_workflow_sags_btn.clicked.connect(self.start_full_workflow_with_sags)
        sags_workflow_layout.addWidget(self.full_workflow_sags_btn)
        
        sags_workflow_group.setLayout(sags_workflow_layout)
        sags_tab_layout.addWidget(sags_workflow_group)
        
        sags_tab_layout.addStretch()
        control_tabs.addTab(sags_tab, "SAGS分割")
        
        # ========== 标签页3: SAGA分割 ==========
        saga_tab = QWidget()
        saga_tab_layout = QVBoxLayout(saga_tab)
        saga_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # SAGA模型加载区域（SAGA使用前也需要加载模型）
        saga_model_group = QGroupBox("模型加载")
        saga_model_layout = QVBoxLayout()
        
        saga_model_layout.addWidget(QLabel("模型路径:"))
        saga_model_path_layout = QHBoxLayout()
        default_saga_model_path = os.path.join(sags_dir, "gaussiansplatting", "output")
        self.saga_model_path_edit = QLineEdit(default_saga_model_path)
        saga_model_path_btn = QPushButton("浏览")
        saga_model_path_btn.clicked.connect(self.saga_browse_model_path)
        saga_model_path_layout.addWidget(self.saga_model_path_edit)
        saga_model_path_layout.addWidget(saga_model_path_btn)
        saga_model_layout.addLayout(saga_model_path_layout)
        
        saga_model_layout.addWidget(QLabel("数据路径:"))
        saga_source_path_layout = QHBoxLayout()
        default_saga_source_path = os.path.join(sags_dir, "gaussiansplatting", "input")
        self.saga_source_path_edit = QLineEdit(default_saga_source_path)
        saga_source_path_btn = QPushButton("浏览")
        saga_source_path_btn.clicked.connect(self.saga_browse_source_path)
        saga_source_path_layout.addWidget(self.saga_source_path_edit)
        saga_source_path_layout.addWidget(saga_source_path_btn)
        saga_model_layout.addLayout(saga_source_path_layout)
        
        saga_model_layout.addWidget(QLabel("迭代次数:"))
        self.saga_iteration_edit = QLineEdit("7000")
        saga_model_layout.addWidget(self.saga_iteration_edit)
        
        saga_model_layout.addWidget(QLabel("分辨率:"))
        self.saga_resolution_combo = QComboBox()
        self.saga_resolution_combo.addItems(["1", "2", "4", "8", "-1"])
        self.saga_resolution_combo.setCurrentText("4")
        saga_model_layout.addWidget(self.saga_resolution_combo)
        
        saga_load_model_btn = QPushButton("加载模型")
        saga_load_model_btn.clicked.connect(self.saga_load_model)
        saga_model_layout.addWidget(saga_load_model_btn)
        
        saga_model_group.setLayout(saga_model_layout)
        saga_tab_layout.addWidget(saga_model_group)
        
        # SAGA分割区域（如果可用）
        if SAGA_AVAILABLE:
            saga_group = QGroupBox("SAGA分割 (SegAnyGAussians)")
            saga_layout = QVBoxLayout()
            
            saga_info_label = QLabel("SAGA是另一种分割方法，基于对比学习特征")
            saga_info_label.setWordWrap(True)
            saga_info_label.setStyleSheet("color: #B0B0B0;")
            saga_layout.addWidget(saga_info_label)
            
            # 数据准备区域
            saga_prepare_group = QGroupBox("数据准备")
            saga_prepare_layout = QVBoxLayout()
            
            saga_prepare_layout.addWidget(QLabel("SAM模型:"))
            self.saga_sam_model_combo = QComboBox()
            self.saga_sam_model_combo.addItems(["vit_h", "vit_l", "vit_b"])
            self.saga_sam_model_combo.setCurrentText("vit_h")
            saga_prepare_layout.addWidget(self.saga_sam_model_combo)
            
            prepare_params_layout = QHBoxLayout()
            prepare_left = QVBoxLayout()
            
            prepare_left.addWidget(QLabel("下采样:"))
            self.saga_downsample_edit = QLineEdit("1")
            prepare_left.addWidget(self.saga_downsample_edit)
            
            prepare_left.addWidget(QLabel("最大长边(内存优化):"))
            self.saga_max_long_side_combo = QComboBox()
            self.saga_max_long_side_combo.addItems(["512", "768", "1024", "1280", "1920"])
            self.saga_max_long_side_combo.setCurrentText("1024")
            prepare_left.addWidget(self.saga_max_long_side_combo)
            
            prepare_right = QVBoxLayout()
            prepare_right.addWidget(QLabel("下采样类型:"))
            self.saga_downsample_type_combo = QComboBox()
            self.saga_downsample_type_combo.addItems(["image", "mask"])
            self.saga_downsample_type_combo.setCurrentText("image")
            prepare_right.addWidget(self.saga_downsample_type_combo)
            
            prepare_params_layout.addLayout(prepare_left)
            prepare_params_layout.addLayout(prepare_right)
            saga_prepare_layout.addLayout(prepare_params_layout)
            
            prepare_buttons_layout = QHBoxLayout()
            saga_extract_btn = QPushButton("提取SAM Masks")
            saga_extract_btn.clicked.connect(self.saga_extract_masks)
            saga_scales_btn = QPushButton("计算Mask Scales")
            saga_scales_btn.clicked.connect(self.saga_get_scales)
            saga_prepare_all_btn = QPushButton("一键准备所有数据")
            saga_prepare_all_btn.clicked.connect(self.saga_prepare_all_data)
            prepare_buttons_layout.addWidget(saga_extract_btn)
            prepare_buttons_layout.addWidget(saga_scales_btn)
            prepare_buttons_layout.addWidget(saga_prepare_all_btn)
            saga_prepare_layout.addLayout(prepare_buttons_layout)
            
            saga_prepare_group.setLayout(saga_prepare_layout)
            saga_layout.addWidget(saga_prepare_group)
            
            # 特征模型加载
            saga_feature_layout = QHBoxLayout()
            saga_check_btn = QPushButton("检查特征模型")
            saga_check_btn.clicked.connect(self.saga_check_feature_model)
            saga_load_btn = QPushButton("加载特征模型")
            saga_load_btn.clicked.connect(self.saga_load_feature_model)
            saga_feature_layout.addWidget(saga_check_btn)
            saga_feature_layout.addWidget(saga_load_btn)
            saga_layout.addLayout(saga_feature_layout)
            
            # 训练对比特征
            saga_train_group = QGroupBox("训练对比特征")
            saga_train_layout = QVBoxLayout()
            
            train_params_layout = QHBoxLayout()
            train_left = QVBoxLayout()
            train_left.addWidget(QLabel("迭代次数:"))
            self.saga_train_iterations_edit = QLineEdit("10000")
            train_left.addWidget(self.saga_train_iterations_edit)
            train_left.addWidget(QLabel("采样光线数(内存优化):"))
            self.saga_num_rays_edit = QLineEdit("500")
            train_left.addWidget(self.saga_num_rays_edit)
            mem_tip_label = QLabel("提示: 内存不足时可降低此值")
            mem_tip_label.setStyleSheet("color: #B0B0B0; font-size: 10px;")
            train_left.addWidget(mem_tip_label)
            
            train_right = QVBoxLayout()
            train_right.addWidget(QLabel("平滑K:"))
            self.saga_smooth_k_edit = QLineEdit("16")
            train_right.addWidget(self.saga_smooth_k_edit)
            train_right.addWidget(QLabel("特征学习率:"))
            self.saga_feature_lr_edit = QLineEdit("0.0025")
            train_right.addWidget(self.saga_feature_lr_edit)
            
            train_params_layout.addLayout(train_left)
            train_params_layout.addLayout(train_right)
            saga_train_layout.addLayout(train_params_layout)
            
            saga_train_btn = QPushButton("开始训练对比特征")
            saga_train_btn.clicked.connect(self.saga_train_features)
            saga_train_layout.addWidget(saga_train_btn)
            
            saga_train_group.setLayout(saga_train_layout)
            saga_layout.addWidget(saga_train_group)
            
            # SAGA分割参数
            saga_seg_group = QGroupBox("分割参数")
            saga_seg_layout = QVBoxLayout()
            
            # 添加参数说明
            saga_help_label = QLabel("📌 使用提示：")
            saga_help_text = QLabel(
                "1. 在图像上点击添加点提示（绿色=正点，红色=负点）\n"
                "2. 调整相似度阈值（推荐0.1-0.3，越高越严格）\n"
                "3. 点击'执行3D分割'进行分割\n"
                "4. 如果选中0个点，请降低阈值或查看终端调试信息"
            )
            saga_help_text.setWordWrap(True)
            saga_help_text.setStyleSheet("color: #B0B0B0; font-size: 10px; padding: 5px;")
            saga_seg_layout.addWidget(saga_help_label)
            saga_seg_layout.addWidget(saga_help_text)
            
            saga_seg_params_layout = QHBoxLayout()
            saga_seg_left = QVBoxLayout()
            saga_seg_left.addWidget(QLabel("相似度阈值:"))
            self.saga_score_thresh_edit = QLineEdit("0.2")
            self.saga_score_thresh_edit.setToolTip("相似度阈值范围 -1.0 到 1.0\n建议值：0.1-0.3\n值越高，选择越严格")
            saga_seg_left.addWidget(self.saga_score_thresh_edit)
            saga_seg_left.addWidget(QLabel("3D尺度:"))
            self.saga_scale_edit = QLineEdit("1.0")
            saga_seg_left.addWidget(self.saga_scale_edit)
            
            saga_seg_right = QVBoxLayout()
            saga_seg_right.addWidget(QLabel("点提示类型:"))
            self.saga_point_type_combo = QComboBox()
            self.saga_point_type_combo.addItems(["正点", "负点"])
            self.saga_point_type_combo.setCurrentText("正点")
            self.saga_point_type_combo.currentTextChanged.connect(self.on_saga_point_type_change)
            saga_seg_right.addWidget(self.saga_point_type_combo)
            
            saga_clear_points_btn = QPushButton("清空点提示")
            saga_clear_points_btn.clicked.connect(self.saga_clear_points)
            saga_seg_right.addWidget(saga_clear_points_btn)
            
            saga_seg_params_layout.addLayout(saga_seg_left)
            saga_seg_params_layout.addLayout(saga_seg_right)
            saga_seg_layout.addLayout(saga_seg_params_layout)
            
            saga_seg_group.setLayout(saga_seg_layout)
            saga_layout.addWidget(saga_seg_group)
            
            # SAGA分割操作
            saga_action_group = QGroupBox("SAGA分割操作")
            saga_action_layout = QVBoxLayout()
            
            saga_actions_inner1 = QHBoxLayout()
            saga_run_seg_btn = QPushButton("执行3D分割")
            saga_run_seg_btn.clicked.connect(self.saga_run_segmentation)
            saga_undo_btn = QPushButton("撤销")
            saga_undo_btn.clicked.connect(self.saga_undo)
            saga_actions_inner1.addWidget(saga_run_seg_btn)
            saga_actions_inner1.addWidget(saga_undo_btn)
            saga_action_layout.addLayout(saga_actions_inner1)
            
            saga_actions_inner2 = QHBoxLayout()
            saga_save_seg_btn = QPushButton("保存分割")
            saga_save_seg_btn.clicked.connect(self.saga_save_segmentation)
            saga_clear_seg_btn = QPushButton("清空分割")
            saga_clear_seg_btn.clicked.connect(self.saga_clear_segmentation)
            saga_actions_inner2.addWidget(saga_save_seg_btn)
            saga_actions_inner2.addWidget(saga_clear_seg_btn)
            saga_action_layout.addLayout(saga_actions_inner2)
            
            saga_action_group.setLayout(saga_action_layout)
            saga_layout.addWidget(saga_action_group)
            
            # SAGA可视化选项
            saga_viz_group = QGroupBox("可视化选项")
            saga_viz_layout = QVBoxLayout()
            
            self.saga_viz_mode_group = QButtonGroup()
            viz_modes = ["RGB", "PCA特征", "相似度图", "3D聚类", "分割结果"]
            for i, mode in enumerate(viz_modes):
                radio = QRadioButton(mode)
                if mode == "RGB":
                    radio.setChecked(True)
                self.saga_viz_mode_group.addButton(radio, i)
                radio.toggled.connect(self.saga_update_visualization)
                saga_viz_layout.addWidget(radio)
            
            saga_viz_group.setLayout(saga_viz_layout)
            saga_layout.addWidget(saga_viz_group)
            
            # 3D聚类参数
            saga_cluster_group = QGroupBox("3D聚类")
            saga_cluster_layout = QVBoxLayout()
            
            cluster_params_layout = QHBoxLayout()
            cluster_left = QVBoxLayout()
            cluster_left.addWidget(QLabel("最小聚类大小:"))
            self.saga_min_cluster_edit = QLineEdit("50")
            cluster_left.addWidget(self.saga_min_cluster_edit)
            
            cluster_right = QVBoxLayout()
            cluster_right.addWidget(QLabel("最小样本数:"))
            self.saga_min_samples_edit = QLineEdit("10")
            cluster_right.addWidget(self.saga_min_samples_edit)
            
            cluster_params_layout.addLayout(cluster_left)
            cluster_params_layout.addLayout(cluster_right)
            saga_cluster_layout.addLayout(cluster_params_layout)
            
            saga_cluster_btn = QPushButton("执行3D聚类")
            saga_cluster_btn.clicked.connect(self.saga_run_clustering)
            saga_cluster_layout.addWidget(saga_cluster_btn)
            
            saga_cluster_group.setLayout(saga_cluster_layout)
            saga_layout.addWidget(saga_cluster_group)
            
            # SAGA状态显示
            self.saga_status_label = QLabel("未加载特征模型")
            self.saga_status_label.setStyleSheet("color: #5BA3D8;")
            saga_layout.addWidget(self.saga_status_label)
            
            saga_group.setLayout(saga_layout)
            saga_tab_layout.addWidget(saga_group)
            saga_tab_layout.addStretch()
            control_tabs.addTab(saga_tab, "SAGA分割")
        else:
            # 如果SAGA不可用，仍然创建标签页但显示提示信息
            no_saga_label = QLabel("SAGA模块不可用")
            no_saga_label.setAlignment(Qt.AlignCenter)
            no_saga_label.setStyleSheet("color: #B0B0B0; font-size: 14px;")
            saga_tab_layout.addWidget(no_saga_label)
            saga_tab_layout.addStretch()
            control_tabs.addTab(saga_tab, "SAGA分割")
        
        # ========== 标签页4: 工具 ==========
        tools_tab = QWidget()
        tools_tab_layout = QVBoxLayout(tools_tab)
        tools_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # 系统管理区域
        system_group = QGroupBox("系统管理")
        system_layout = QVBoxLayout()
        
        reset_system_btn = QPushButton("🔄 重置系统")
        reset_system_btn.setStyleSheet("""
            QPushButton {
                background-color: #C75050;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #D96A6A;
            }
            QPushButton:pressed {
                background-color: #B53838;
            }
        """)
        reset_system_btn.clicked.connect(self.reset_system)
        system_layout.addWidget(reset_system_btn)
        
        reset_info_label = QLabel("重置系统到初始状态，清除所有数据和进程")
        reset_info_label.setWordWrap(True)
        reset_info_label.setStyleSheet("color: #B0B0B0; font-size: 10px;")
        system_layout.addWidget(reset_info_label)
        
        system_group.setLayout(system_layout)
        tools_tab_layout.addWidget(system_group)
        
        # 内存管理区域
        memory_group = QGroupBox("内存管理")
        memory_layout = QVBoxLayout()
        
        clear_memory_btn = QPushButton("清理GPU内存")
        clear_memory_btn.clicked.connect(self.clear_gpu_memory)
        memory_layout.addWidget(clear_memory_btn)
        
        show_memory_btn = QPushButton("显示内存状态")
        show_memory_btn.clicked.connect(self.show_memory_status)
        memory_layout.addWidget(show_memory_btn)
        
        memory_group.setLayout(memory_layout)
        tools_tab_layout.addWidget(memory_group)
        
        tools_tab_layout.addStretch()
        control_tabs.addTab(tools_tab, "工具")
        
        # ========== 标签页5: GIS视图 ==========
        if GIS_AVAILABLE:
            gis_tab = QWidget()
            gis_tab_layout = QVBoxLayout(gis_tab)
            gis_tab_layout.setContentsMargins(5, 5, 5, 5)
            
            # GIS配置组
            gis_config_group = QGroupBox("地理坐标配置")
            gis_config_layout = QVBoxLayout()
            
            # 预设位置选择
            preset_layout = QHBoxLayout()
            preset_layout.addWidget(QLabel("预设位置:"))
            self.gis_preset_combo = QComboBox()
            self.gis_preset_combo.addItems(["自定义"] + list(PRESET_LOCATIONS.keys()))
            self.gis_preset_combo.currentTextChanged.connect(self.on_gis_preset_changed)
            preset_layout.addWidget(self.gis_preset_combo)
            gis_config_layout.addLayout(preset_layout)
            
            # 原点经度
            lon_layout = QHBoxLayout()
            lon_layout.addWidget(QLabel("经度:"))
            self.gis_longitude_input = QDoubleSpinBox()
            self.gis_longitude_input.setRange(-180.0, 180.0)
            self.gis_longitude_input.setDecimals(6)
            self.gis_longitude_input.setValue(114.610945)  # 默认坐标
            self.gis_longitude_input.setSuffix("°")
            lon_layout.addWidget(self.gis_longitude_input)
            gis_config_layout.addLayout(lon_layout)
            
            # 原点纬度
            lat_layout = QHBoxLayout()
            lat_layout.addWidget(QLabel("纬度:"))
            self.gis_latitude_input = QDoubleSpinBox()
            self.gis_latitude_input.setRange(-90.0, 90.0)
            self.gis_latitude_input.setDecimals(6)
            self.gis_latitude_input.setValue(30.457906)  # 默认坐标
            self.gis_latitude_input.setSuffix("°")
            lat_layout.addWidget(self.gis_latitude_input)
            gis_config_layout.addLayout(lat_layout)
            
            # 原点高度
            alt_layout = QHBoxLayout()
            alt_layout.addWidget(QLabel("高度:"))
            self.gis_altitude_input = QDoubleSpinBox()
            self.gis_altitude_input.setRange(-1000.0, 10000.0)
            self.gis_altitude_input.setDecimals(2)
            self.gis_altitude_input.setValue(0.0)
            self.gis_altitude_input.setSuffix(" m")
            alt_layout.addWidget(self.gis_altitude_input)
            gis_config_layout.addLayout(alt_layout)
            
            # 缩放比例
            scale_layout = QHBoxLayout()
            scale_layout.addWidget(QLabel("缩放:"))
            self.gis_scale_input = QDoubleSpinBox()
            self.gis_scale_input.setRange(0.001, 1000.0)
            self.gis_scale_input.setDecimals(3)
            self.gis_scale_input.setValue(1.0)
            self.gis_scale_input.setToolTip("本地坐标单位对应的米数")
            scale_layout.addWidget(self.gis_scale_input)
            gis_config_layout.addLayout(scale_layout)
            
            # 旋转角度
            rotation_layout = QHBoxLayout()
            rotation_layout.addWidget(QLabel("旋转:"))
            self.gis_rotation_input = QDoubleSpinBox()
            self.gis_rotation_input.setRange(-180.0, 180.0)
            self.gis_rotation_input.setDecimals(2)
            self.gis_rotation_input.setValue(0.0)
            self.gis_rotation_input.setSuffix("°")
            rotation_layout.addWidget(self.gis_rotation_input)
            gis_config_layout.addLayout(rotation_layout)
            
            # 自动地理定位
            auto_geo_layout = QHBoxLayout()
            self.gis_auto_geo_checkbox = QCheckBox("自动检测地理位置（从图像EXIF）")
            self.gis_auto_geo_checkbox.setChecked(True)
            self.gis_auto_geo_checkbox.setToolTip("从训练图像的EXIF数据中自动检测GPS坐标")
            auto_geo_layout.addWidget(self.gis_auto_geo_checkbox)
            gis_config_layout.addLayout(auto_geo_layout)
            
            # 保存/加载配置按钮
            config_btn_layout = QHBoxLayout()
            self.gis_save_config_btn = QPushButton("保存配置")
            self.gis_save_config_btn.clicked.connect(self.save_gis_config)
            self.gis_load_config_btn = QPushButton("加载配置")
            self.gis_load_config_btn.clicked.connect(self.load_gis_config)
            config_btn_layout.addWidget(self.gis_save_config_btn)
            config_btn_layout.addWidget(self.gis_load_config_btn)
            gis_config_layout.addLayout(config_btn_layout)
            
            gis_config_group.setLayout(gis_config_layout)
            gis_tab_layout.addWidget(gis_config_group)
            
            # GIS导出组
            gis_export_group = QGroupBox("模型导出")
            gis_export_layout = QVBoxLayout()
            
            # 导出格式选择
            export_format_layout = QHBoxLayout()
            export_format_layout.addWidget(QLabel("格式:"))
            self.gis_export_format_combo = QComboBox()
            self.gis_export_format_combo.addItems(["点云JSON", "3D Tiles"])
            export_format_layout.addWidget(self.gis_export_format_combo)
            gis_export_layout.addLayout(export_format_layout)
            
            # 采样率
            sample_layout = QHBoxLayout()
            sample_layout.addWidget(QLabel("采样率:"))
            self.gis_sample_rate_slider = QSlider(Qt.Horizontal)
            self.gis_sample_rate_slider.setRange(1, 100)
            self.gis_sample_rate_slider.setValue(100)
            self.gis_sample_rate_label = QLabel("100%")
            self.gis_sample_rate_slider.valueChanged.connect(
                lambda v: self.gis_sample_rate_label.setText(f"{v}%")
            )
            sample_layout.addWidget(self.gis_sample_rate_slider)
            sample_layout.addWidget(self.gis_sample_rate_label)
            gis_export_layout.addLayout(sample_layout)
            
            # 透明度阈值
            opacity_layout = QHBoxLayout()
            opacity_layout.addWidget(QLabel("透明度阈值:"))
            self.gis_opacity_threshold_slider = QSlider(Qt.Horizontal)
            self.gis_opacity_threshold_slider.setRange(0, 100)
            self.gis_opacity_threshold_slider.setValue(10)
            self.gis_opacity_threshold_label = QLabel("0.10")
            self.gis_opacity_threshold_slider.valueChanged.connect(
                lambda v: self.gis_opacity_threshold_label.setText(f"{v/100:.2f}")
            )
            opacity_layout.addWidget(self.gis_opacity_threshold_slider)
            opacity_layout.addWidget(self.gis_opacity_threshold_label)
            gis_export_layout.addLayout(opacity_layout)
            
            # 导出按钮
            self.gis_export_btn = QPushButton("导出模型")
            self.gis_export_btn.clicked.connect(self.export_model_to_gis)
            self.gis_export_btn.setEnabled(False)
            gis_export_layout.addWidget(self.gis_export_btn)
            
            gis_export_group.setLayout(gis_export_layout)
            gis_tab_layout.addWidget(gis_export_group)
            
            # GIS加载组
            gis_load_group = QGroupBox("Cesium加载")
            gis_load_layout = QVBoxLayout()
            
            # 加载到Cesium按钮
            self.gis_load_to_cesium_btn = QPushButton("加载当前模型到Cesium")
            self.gis_load_to_cesium_btn.clicked.connect(self.load_model_to_cesium)
            self.gis_load_to_cesium_btn.setEnabled(False)
            gis_load_layout.addWidget(self.gis_load_to_cesium_btn)
            
            # 加载本地文件到Cesium按钮
            self.gis_load_local_file_btn = QPushButton("加载本地模型文件(.ply/.splat)")
            self.gis_load_local_file_btn.clicked.connect(self.load_local_model_to_cesium)
            gis_load_layout.addWidget(self.gis_load_local_file_btn)
            
            # 清除Cesium按钮
            self.gis_clear_cesium_btn = QPushButton("清除Cesium场景")
            self.gis_clear_cesium_btn.clicked.connect(self.clear_cesium_scene)
            gis_load_layout.addWidget(self.gis_clear_cesium_btn)
            
            gis_load_group.setLayout(gis_load_layout)
            gis_tab_layout.addWidget(gis_load_group)
            
            # 手势控制组
            gis_gesture_group = QGroupBox("手势控制")
            gis_gesture_layout = QVBoxLayout()
            
            # 启用按钮
            self.gis_gesture_btn = QPushButton("启用手势控制")
            gis_gesture_layout.addWidget(self.gis_gesture_btn)
            
            # 摄像头预览
            self.gis_camera_label = QLabel()
            self.gis_camera_label.setFixedSize(320, 240)
            self.gis_camera_label.setStyleSheet("background-color: black; border: 1px solid #555;")
            self.gis_camera_label.setAlignment(Qt.AlignCenter)
            self.gis_camera_label.hide() # 默认隐藏
            
            # 居中显示摄像头画面
            camera_container = QHBoxLayout()
            camera_container.addStretch()
            camera_container.addWidget(self.gis_camera_label)
            camera_container.addStretch()
            gis_gesture_layout.addLayout(camera_container)
            
            gis_gesture_group.setLayout(gis_gesture_layout)
            gis_tab_layout.addWidget(gis_gesture_group)
            
            # 状态显示
            self.gis_status_label = QLabel("GIS模块已就绪")
            self.gis_status_label.setStyleSheet("color: #5BA3D8; padding: 5px;")
            gis_tab_layout.addWidget(self.gis_status_label)
            
            gis_tab_layout.addStretch()
            control_tabs.addTab(gis_tab, "GIS视图")
        else:
            # 如果GIS不可用，创建提示标签页
            gis_tab = QWidget()
            gis_tab_layout = QVBoxLayout(gis_tab)
            no_gis_label = QLabel("GIS模块不可用\n请安装PyQtWebEngine")
            no_gis_label.setAlignment(Qt.AlignCenter)
            no_gis_label.setStyleSheet("color: #B0B0B0; font-size: 14px;")
            gis_tab_layout.addWidget(no_gis_label)
            gis_tab_layout.addStretch()
            control_tabs.addTab(gis_tab, "GIS视图")
        
        # 将所有标签页添加到父布局
        parent_layout.addWidget(control_tabs)
    
    def create_display_panel(self, parent_layout):
        """创建中间显示面板"""
        # 创建标签页
        self.display_tabs = QTabWidget()
        
        # 3D渲染标签页（包含编辑功能）- 类DIVSHOT专业编辑器风格
        self.render_widget = QWidget()
        render_layout = QVBoxLayout(self.render_widget)
        render_layout.setContentsMargins(0, 0, 0, 0)
        render_layout.setSpacing(0)
        
        # 顶部工具栏 - 只保留SIBR控制
        top_toolbar = QWidget()
        top_toolbar.setFixedHeight(48)
        top_toolbar.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border-bottom: 1px solid #404040;
            }
        """)
        toolbar_layout = QHBoxLayout(top_toolbar)
        toolbar_layout.setContentsMargins(10, 6, 10, 6)
        toolbar_layout.setSpacing(8)
        
        # 分割对比按钮
        self.comparison_btn = QPushButton("显示分割后模型")
        self.comparison_btn.setMinimumHeight(32)
        self.comparison_btn.setCheckable(True)
        self.comparison_btn.setEnabled(False)  # 初始禁用，分割完成后启用
        self.comparison_btn.clicked.connect(self.toggle_segmentation_comparison)
        self.comparison_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 4px 12px;
                color: #B0B0B0;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
                border-color: #606060;
            }
            QPushButton:checked {
                background-color: #0E639C;
                border-color: #0E639C;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                background-color: #2B2B2B;
                color: #606060;
            }
        """)
        toolbar_layout.addWidget(self.comparison_btn)
        
        # SIBR查看器控制
        launch_viewer_btn = QPushButton("启动SIBR")
        launch_viewer_btn.setMinimumHeight(32)
        launch_viewer_btn.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        launch_viewer_btn.clicked.connect(self.launch_3d_viewer)
        toolbar_layout.addWidget(launch_viewer_btn)
        
        stop_viewer_btn = QPushButton("停止")
        stop_viewer_btn.setMinimumHeight(32)
        stop_viewer_btn.setStyleSheet("""
            QPushButton {
                background-color: #C75050;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #D96A6A;
            }
        """)
        stop_viewer_btn.clicked.connect(self.stop_3d_viewer)
        toolbar_layout.addWidget(stop_viewer_btn)
        
        toolbar_layout.addStretch()
        
        render_layout.addWidget(top_toolbar)
        
        # 主视图区域（左侧工具栏 + 3D视图 + 右侧属性面板）- DIVSHOT风格
        main_view_widget = QWidget()
        main_view_layout = QHBoxLayout(main_view_widget)
        main_view_layout.setContentsMargins(0, 0, 0, 0)
        main_view_layout.setSpacing(0)
        
        # 3D编辑视图（如果编辑器可用）或提示信息
        if EDITOR_AVAILABLE:
            # 左侧工具栏 - DIVSHOT风格的图标化工具栏
            self.left_toolbar = QWidget()
            self.left_toolbar.setFixedWidth(56)
            self.left_toolbar.setStyleSheet("""
                QWidget {
                    background-color: #252525;
                    border-right: 1px solid #404040;
                }
            """)
            left_toolbar_layout = QVBoxLayout(self.left_toolbar)
            left_toolbar_layout.setContentsMargins(4, 8, 4, 8)
            left_toolbar_layout.setSpacing(4)
            left_toolbar_layout.setAlignment(Qt.AlignTop)
            
            # 创建图标化的工具按钮
            self.create_icon_toolbar(left_toolbar_layout)
            
            main_view_layout.addWidget(self.left_toolbar)
            
            # 中间3D视图
            self.edit_3d_viewer = Gaussian3DViewer()
            self.edit_3d_viewer.selection_changed.connect(self.on_edit_selection_changed)
            main_view_layout.addWidget(self.edit_3d_viewer, stretch=1)
            
            # 右侧属性面板（编辑模式下显示，可折叠）
            self.right_edit_panel = QWidget()
            self.right_edit_panel.setFixedWidth(320)
            self.right_edit_panel.setStyleSheet("""
                QWidget {
                    background-color: #252525;
                    border-left: 1px solid #404040;
                }
            """)
            self.right_edit_panel.hide()  # 默认隐藏
            
            edit_panel_layout = QVBoxLayout(self.right_edit_panel)
            edit_panel_layout.setContentsMargins(0, 0, 0, 0)
            edit_panel_layout.setSpacing(0)
            
            # 使用标签页组织工具 - 更简洁清晰
            self.edit_panel_tabs = QTabWidget()
            self.edit_panel_tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: none;
                    background-color: #252525;
                }
                QTabBar::tab {
                    background-color: #2B2B2B;
                    color: #B0B0B0;
                    padding: 8px 16px;
                    border: none;
                    border-bottom: 2px solid transparent;
                }
                QTabBar::tab:selected {
                    color: #FFFFFF;
                    border-bottom: 2px solid #0E639C;
                }
                QTabBar::tab:hover:!selected {
                    background-color: #3A3A3A;
                }
            """)
            edit_panel_layout.addWidget(self.edit_panel_tabs)
            
            # 创建各个标签页
            self.create_transform_and_settings_tab()
            
            main_view_layout.addWidget(self.right_edit_panel)
        else:
            # 提示信息（仅在编辑器不可用时显示）
            info_label = QLabel("使用上方的按钮启动SIBR查看器来查看3D模型")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("""
                QLabel {
                    color: #B0B0B0;
                    font-size: 14px;
                    padding: 20px;
                    background-color: #2B2B2B;
                    border-radius: 5px;
                    border: 1px solid #404040;
                }
            """)
            main_view_layout.addWidget(info_label)
        
        render_layout.addWidget(main_view_widget, stretch=1)
        
        # 底部状态栏
        render_status_bar = QWidget()
        render_status_bar.setFixedHeight(30)
        render_status_bar.setStyleSheet("""
            QWidget {
                background-color: #2B2B2B;
                border-top: 1px solid #404040;
            }
        """)
        render_status_layout = QHBoxLayout(render_status_bar)
        render_status_layout.setContentsMargins(10, 5, 10, 5)
        render_status_layout.setSpacing(10)
        
        # 状态文本
        self.render_status_label = QLabel("未加载模型")
        self.render_status_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        render_status_layout.addWidget(self.render_status_label)
        render_status_layout.addStretch()
        
        render_layout.addWidget(render_status_bar)
        
        self.display_tabs.addTab(self.render_widget, "3D渲染")
        
        # SAGS分割标签页（合并原始图像和分割结果）
        sags_split_widget = QWidget()
        sags_split_layout = QHBoxLayout(sags_split_widget)
        sags_split_layout.setContentsMargins(5, 5, 5, 5)
        sags_split_layout.setSpacing(5)
        
        # 左侧：SAGS分割（原原始图像）
        left_group = QGroupBox("SAGS分割")
        left_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2B2B2B;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2B2B2B;
                color: #E0E0E0;
            }
        """)
        left_layout = QVBoxLayout(left_group)
        self.original_canvas = ImageCanvas()
        self.original_canvas.clicked.connect(lambda x, y: self.on_canvas_click(x, y, "original"))
        left_layout.addWidget(self.original_canvas)
        sags_split_layout.addWidget(left_group, stretch=1)
        
        # 右侧：分割结果
        right_group = QGroupBox("分割结果")
        right_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2B2B2B;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: #2B2B2B;
                color: #E0E0E0;
            }
        """)
        right_layout = QVBoxLayout(right_group)
        self.segmentation_canvas = ImageCanvas()
        right_layout.addWidget(self.segmentation_canvas)
        sags_split_layout.addWidget(right_group, stretch=1)
        
        self.display_tabs.addTab(sags_split_widget, "SAGS分割")
        
        # SAGA分割标签页（如果可用）
        if SAGA_AVAILABLE:
            self.saga_canvas = ImageCanvas()
            self.saga_canvas.clicked.connect(lambda x, y: self.on_saga_canvas_click(x, y))
            self.display_tabs.addTab(self.saga_canvas, "SAGA分割")
        
        # Cesium GIS视图标签页
        if GIS_AVAILABLE:
            self.cesium_panel = CesiumPanel()
            self.cesium_widget = self.cesium_panel.get_cesium_widget()
            
            # 设置手势控制组件
            if hasattr(self, 'gis_gesture_btn') and hasattr(self, 'gis_camera_label'):
                self.cesium_panel.setup_gesture_control(self.gis_gesture_btn, self.gis_camera_label)
            
            # 连接Cesium信号
            self.cesium_widget.viewer_ready.connect(self.on_cesium_viewer_ready)
            self.cesium_widget.load_complete.connect(self.on_cesium_load_complete)
            self.cesium_widget.object_clicked.connect(self.on_cesium_object_clicked)
            self.display_tabs.addTab(self.cesium_panel, "Cesium GIS")
        
        # 连接标签页切换事件，当切换到SAGA标签页时自动更新可视化
        # 注意：即使SAGA不可用也要连接，以便在SAGA可用后可以正常使用
        self.display_tabs.currentChanged.connect(self.on_display_tab_changed)
        
        parent_layout.addWidget(self.display_tabs)
    
    def create_log_panel(self, parent_layout):
        """创建右侧日志面板"""
        log_label = QLabel("系统日志")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        log_label.setFont(font)
        parent_layout.addWidget(log_label)
        
        # 日志控制按钮
        log_control_layout = QHBoxLayout()
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_log)
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_log)
        log_control_layout.addWidget(clear_btn)
        log_control_layout.addWidget(save_btn)
        log_control_layout.addStretch()
        parent_layout.addLayout(log_control_layout)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 11))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #252525;
                color: #E0E0E0;
                border: 1px solid #404040;
            }
        """)
        parent_layout.addWidget(self.log_text)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        
        # 左侧状态信息
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 右侧系统信息
        self.system_info_label = QLabel("")
        self.status_bar.addPermanentWidget(self.system_info_label)
    
    # 以下方法需要实现，我会添加占位符实现
    
    def on_colmap_input_type_change(self):
        """COLMAP输入类型改变"""
        is_video = self.colmap_video_radio.isChecked()
        self.video_params_group.setVisible(is_video)
        if is_video:
            self.update_video_info()
    
    def update_video_info(self):
        """更新视频信息"""
        video_path = self.colmap_input_path_edit.text()
        if video_path and os.path.exists(video_path):
            try:
                info = get_video_info(video_path)
                self.video_info_label.setText(f"视频信息: {info}")
            except Exception as e:
                self.video_info_label.setText(f"无法获取视频信息: {str(e)}")
        else:
            self.video_info_label.setText("")
    
    def browse_colmap_input_path(self):
        """浏览COLMAP输入路径"""
        if self.colmap_images_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "选择图像目录")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.colmap_input_path_edit.setText(path)
            if self.colmap_video_radio.isChecked():
                self.update_video_info()
    
    def browse_original_images(self):
        """浏览并加载原始图像目录"""
        path = QFileDialog.getExistingDirectory(self, "选择原始图像目录")
        if path:
            self.load_original_images(path)
    
    def update_colmap_output_path(self, input_path):
        """更新COLMAP输出路径"""
        # 自动设置输出路径逻辑
        pass
    
    def browse_vocab_tree(self):
        """浏览词汇树文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择词汇树文件", "", "Binary Files (*.bin)")
        if path:
            self.colmap_vocab_tree_edit.setText(path)
    
    def validate_colmap_params(self):
        """验证COLMAP参数"""
        try:
            input_path = self.colmap_input_path_edit.text()
            input_type = "video" if self.colmap_video_radio.isChecked() else "images"
            
            if not input_path:
                raise ValueError("请输入COLMAP输入路径")
            
            # 如果是视频输入，验证视频参数
            if input_type == "video":
                try:
                    frame_rate = int(self.video_frame_rate_edit.text())
                    if frame_rate <= 0:
                        raise ValueError("提取帧率必须大于0")
                    
                    quality = int(self.video_quality_edit.text())
                    if quality < 1 or quality > 100:
                        raise ValueError("图像质量必须在1-100之间")
                    
                    max_frames = int(self.video_max_frames_edit.text())
                    if max_frames < 0:
                        raise ValueError("最大帧数不能为负数")
                    
                    resize_width = int(self.video_resize_width_edit.text())
                    if resize_width < 0:
                        raise ValueError("调整宽度不能为负数")
                except ValueError as e:
                    if "必须" in str(e) or "必须" in str(e) or "不能" in str(e):
                        raise e
                    raise ValueError("视频参数格式错误")
            
            camera_model = self.colmap_camera_model_combo.currentText()
            if camera_model not in ["OPENCV", "PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                raise ValueError("无效的相机模型")
            
            return True
        except ValueError as e:
            self.show_message("参数错误", str(e), "critical")
            return False
    
    def start_colmap_processing(self):
        """开始COLMAP处理"""
        if self.is_colmap_processing:
            self.show_message("警告", "COLMAP正在处理中", "warning")
            return
        
        # 验证COLMAP参数
        if not self.validate_colmap_params():
            return
        
        input_path = self.colmap_input_path_edit.text()
        input_type = "video" if self.colmap_video_radio.isChecked() else "images"
        
        if not input_path or not os.path.exists(input_path):
            self.show_message("错误", "请选择有效的COLMAP输入路径", "critical")
            return
        
        # 根据输入类型验证
        if input_type == "video":
            # 验证视频文件
            is_valid, message = validate_video_file(input_path)
            if not is_valid:
                self.show_message("错误", f"视频文件验证失败: {message}", "critical")
                return
            
            # 获取视频文件名（不含扩展名）作为文件夹名
            video_basename = os.path.splitext(os.path.basename(input_path))[0]
            folder_name = video_basename
        else:
            # 验证图像目录
            is_valid, message = validate_image_directory(input_path)
            if not is_valid:
                self.show_message("错误", f"图像目录验证失败: {message}", "critical")
                return
            
            # 获取输入文件夹名
            folder_name = os.path.basename(input_path.rstrip('/'))
        
        # 自动设置输出路径
        sags_dir = os.path.dirname(__file__)
        output_path = os.path.join(sags_dir, "gaussiansplatting", "input", folder_name)
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        def colmap_thread():
            try:
                self.is_colmap_processing = True
                self.colmap_status_signal.emit("正在处理...")
                self.update_status("正在执行COLMAP处理...", 0)
                self.set_enabled_signal.emit(self.start_colmap_btn, False)
                self.set_enabled_signal.emit(self.stop_colmap_btn, True)
                
                # 保存COLMAP参数
                colmap_input_path = input_path
                colmap_input_type = input_type
                colmap_output_path = output_path
                camera_model = self.colmap_camera_model_combo.currentText()
                single_camera = self.colmap_single_camera_check.isChecked()
                quality = self.colmap_quality_combo.currentText()
                data_type = self.colmap_data_type_combo.currentText()
                mapper_type = self.colmap_mapper_type_combo.currentText()
                num_threads = self.colmap_num_threads_edit.text()
                sparse_model = self.colmap_sparse_model_check.isChecked()
                dense_model = self.colmap_dense_model_check.isChecked()
                use_gpu = self.colmap_use_gpu_check.isChecked()
                vocab_tree = self.colmap_vocab_tree_edit.text()
                
                # 创建COLMAP处理器
                try:
                    self.colmap_processor = COLMAPProcessor()
                    self.log(f"COLMAP路径: {self.colmap_processor.colmap_path}")
                except FileNotFoundError as e:
                    self.log(f"COLMAP处理器初始化失败: {str(e)}", "error")
                    self.show_message("错误", f"COLMAP处理器初始化失败: {str(e)}", "critical")
                    return
                
                # 设置进度回调
                def progress_callback(message, progress=None):
                    if progress is not None:
                        self.update_status(f"COLMAP: {message}", progress)
                        self.log(f"[COLMAP {progress:.0f}%] {message}")
                    else:
                        self.log(f"[COLMAP] {message}")
                
                self.colmap_processor.set_progress_callback(progress_callback)
                
                # 如果输入是视频，先提取帧
                actual_image_dir = colmap_input_path
                if colmap_input_type == "video":
                    self.log("开始从视频提取图像帧...")
                    self.log(f"视频文件: {colmap_input_path}")
                    
                    # 创建临时图像目录
                    temp_images_dir = os.path.join(colmap_output_path, "extracted_frames")
                    os.makedirs(temp_images_dir, exist_ok=True)
                    
                    # 获取视频参数
                    try:
                        frame_rate = int(self.video_frame_rate_edit.text())
                        quality_val = int(self.video_quality_edit.text())
                        max_frames = int(self.video_max_frames_edit.text())
                        resize_width = int(self.video_resize_width_edit.text())
                        
                        if max_frames == 0:
                            max_frames = None
                        if resize_width == 0:
                            resize_width = None
                            
                        self.log(f"提取参数: 帧率={frame_rate}fps, 质量={quality_val}, 最大帧数={max_frames or '无限制'}, 调整宽度={resize_width or '原始'}")
                    except ValueError as e:
                        self.log(f"视频参数错误: {str(e)}", "error")
                        self.show_message("错误", f"视频参数错误: {str(e)}", "critical")
                        return
                    
                    # 提取视频帧
                    success, message, frame_count = self.colmap_processor.extract_frames_from_video(
                        video_path=colmap_input_path,
                        output_dir=temp_images_dir,
                        frame_rate=frame_rate,
                        quality=quality_val,
                        max_frames=max_frames,
                        resize_width=resize_width
                    )
                    
                    if not success:
                        self.log(f"视频帧提取失败: {message}", "error")
                        self.show_message("错误", f"视频帧提取失败: {message}", "critical")
                        return
                    
                    self.log(f"视频帧提取成功: {message}")
                    actual_image_dir = temp_images_dir
                
                # 开始COLMAP处理
                self.log("开始COLMAP处理...")
                self.log(f"输入图像路径: {actual_image_dir}")
                self.log(f"输出路径: {colmap_output_path}")
                self.log(f"相机模型: {camera_model}")
                self.log(f"单相机模式: {single_camera}")
                self.log(f"处理质量: {quality}")
                self.log(f"数据类型: {data_type}")
                self.log(f"映射器类型: {mapper_type}")
                self.log(f"线程数: {num_threads}")
                self.log(f"稀疏模型: {sparse_model}")
                self.log(f"密集模型: {dense_model}")
                self.log(f"GPU加速: {use_gpu}")
                if vocab_tree and os.path.exists(vocab_tree):
                    self.log(f"词汇树文件: {vocab_tree}")
                
                # 使用COLMAP自动重建
                success = self.colmap_processor.auto_reconstruction(
                    image_dir=actual_image_dir,
                    workspace_path=colmap_output_path,
                    camera_model=camera_model,
                    single_camera=single_camera,
                    quality=quality,
                    data_type=data_type,
                    mapper_type=mapper_type,
                    num_threads=num_threads,
                    sparse_model=sparse_model,
                    dense_model=dense_model,
                    use_gpu=use_gpu,
                    vocab_tree=vocab_tree if vocab_tree and os.path.exists(vocab_tree) else None
                )
                
                if success:
                    self.update_status("COLMAP处理完成", 100)
                    self.log("COLMAP处理成功完成！")
                    self.colmap_status_signal.emit("处理完成")
                    
                    # 自动更新训练数据路径
                    self.set_text_signal.emit(self.training_data_path_edit, colmap_output_path)
                    self.log(f"已自动更新训练数据路径: {colmap_output_path}")
                    
                    # 自动更新SOG训练数据路径
                    self.set_text_signal.emit(self.sog_data_path_edit, colmap_output_path)
                    self.log(f"已自动更新SOG训练数据路径: {colmap_output_path}")
                    
                    # 更新系统信息
                    self.update_system_info()
                    
                    # 显示处理完成信息（使用信号在主线程中显示）
                    completion_msg = f"""COLMAP处理完成！

输入路径: {colmap_input_path}
输出路径: {colmap_output_path}

是否立即开始3DGS训练？"""
                    
                    # 使用QTimer.singleShot在主线程中显示对话框
                    def show_dialog():
                        reply = QMessageBox.question(self, "COLMAP处理完成", completion_msg, 
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            # 自动更新训练数据路径
                            self.set_text_signal.emit(self.training_data_path_edit, colmap_output_path)
                            self.log("已自动更新训练数据路径")
                    QTimer.singleShot(0, show_dialog)
                else:
                    self.update_status("COLMAP处理失败", 0)
                    self.log("COLMAP处理失败", "error")
                    self.colmap_status_signal.emit("处理失败")
                    self.show_message("错误", "COLMAP处理失败，请检查日志", "critical")
                
            except Exception as e:
                self.update_status("COLMAP处理出错", 0)
                self.log(f"COLMAP处理过程中发生错误: {str(e)}", "error")
                self.colmap_status_signal.emit("处理出错")
                self.show_message("错误", f"COLMAP处理失败: {str(e)}", "critical")
            finally:
                self.is_colmap_processing = False
                self.set_enabled_signal.emit(self.start_colmap_btn, True)
                self.set_enabled_signal.emit(self.stop_colmap_btn, False)
                self.colmap_processor = None
        
        # 在后台线程中运行COLMAP处理
        self.colmap_thread = threading.Thread(target=colmap_thread, daemon=True)
        self.colmap_thread.start()
    
    def stop_colmap_processing(self):
        """停止COLMAP处理"""
        if not self.is_colmap_processing or not self.colmap_processor:
            return
        
        try:
            self.log("正在停止COLMAP处理...")
            self.colmap_processor.stop_processing()
            
            self.log("COLMAP处理已停止")
            self.colmap_status_signal.emit("处理已停止")
            
        except Exception as e:
            self.log(f"停止COLMAP处理时发生错误: {str(e)}")
        finally:
            self.is_colmap_processing = False
            self.set_enabled_signal.emit(self.start_colmap_btn, True)
            self.set_enabled_signal.emit(self.stop_colmap_btn, False)
            self.colmap_processor = None
    
    def on_canvas_click(self, x, y, canvas_type):
        """画布点击事件"""
        # 检查是否启用预处理且当前显示的是原始图像
        if (canvas_type == "original" and 
            self.enable_preprocess_check.isChecked() and 
            (self.preprocess_point_prompt_radio.isChecked() if hasattr(self, 'preprocess_point_prompt_radio') else False) and 
            self.original_images and 
            self.current_image is not None):
            
            # 处理预处理点点击
            self.on_preprocess_canvas_click(x, y)
            return
        
        # 原有的模型加载后的点击处理
        if canvas_type == "original":
            canvas = self.original_canvas
            # 只有在模型已加载且使用点提示时才处理
            if not self.model_loaded or not self.point_prompt_radio.isChecked():
                return
        else:
            canvas = self.segmentation_canvas
        
        # 根据当前点类型添加点
        label = self.current_point_type
        canvas.add_point(x, y, label)
        self.click_points = canvas.click_points
        self.update_points_display()
        self.log(f"添加点: ({x}, {y}), 类型: {'正点' if label == 1 else '负点'}", "info")
    
    def on_saga_canvas_click(self, x, y):
        """SAGA画布点击事件"""
        label = self.saga_current_point_type
        self.saga_canvas.add_point(x, y, label)
        self.saga_point_prompts = self.saga_canvas.click_points
    
    def log(self, message, level="info"):
        """添加日志（线程安全）"""
        # 使用信号确保在主线程中更新UI
        self.log_signal.emit(message, level)
    
    def _log_internal(self, message, level="info"):
        """内部日志方法，在主线程中调用"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 根据级别选择颜色
        color_map = {
            "info": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "debug": "#6c757d"
        }
        color = color_map.get(level, "#212529")
        
        formatted_message = f'<span style="color: {color}">[{timestamp}] [{level.upper()}] {message}</span>'
        self.log_text.append(formatted_message)
        
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
    
    def save_log(self):
        """保存日志"""
        path, _ = QFileDialog.getSaveFileName(self, "保存日志", "", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log(f"日志已保存到: {path}", "info")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存日志失败: {str(e)}")
    
    def generate_mesh(self):
        """生成Mesh（带参数设置对话框）"""
        if not MULTI_MODE_AVAILABLE:
            self.show_message("错误", "多模式渲染器不可用", "critical")
            return
        
        if not hasattr(self, 'edit_3d_viewer') or self.edit_3d_viewer is None:
            self.show_message("提示", "请先加载模型并打开3D视图", "warning")
            return
        
        # 创建参数设置对话框
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Mesh生成参数设置")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # 重建方法
        method_group = QGroupBox("重建方法")
        method_layout = QVBoxLayout()
        
        method_combo = QComboBox()
        method_combo.addItems([
            "Poisson表面重建（推荐，高质量）",
            "Alpha Shape（保留细节）",
            "Ball Pivoting（快速）"
        ])
        method_combo.setCurrentIndex(0)
        
        method_label = QLabel("• Poisson: 生成光滑、水密的表面\n• Alpha Shape: 保留更多细节但可能有孔洞\n• Ball Pivoting: 快速但质量较低")
        method_label.setStyleSheet("color: #808080; font-size: 10px;")
        method_label.setWordWrap(True)
        
        method_layout.addWidget(QLabel("选择方法:"))
        method_layout.addWidget(method_combo)
        method_layout.addWidget(method_label)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # Poisson参数
        poisson_group = QGroupBox("Poisson参数（仅对Poisson方法有效）")
        poisson_layout = QVBoxLayout()
        
        # Depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("重建深度 (6-12):"))
        depth_spin = QSpinBox()
        depth_spin.setRange(6, 12)
        depth_spin.setValue(10)
        depth_spin.setToolTip("越高越精细，但计算时间更长。推荐: 9-10")
        depth_layout.addWidget(depth_spin)
        poisson_layout.addLayout(depth_layout)
        
        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放因子 (1.0-1.5):"))
        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(1.0, 1.5)
        scale_spin.setValue(1.1)
        scale_spin.setSingleStep(0.05)
        scale_spin.setToolTip("控制表面外推程度。推荐: 1.1")
        scale_layout.addWidget(scale_spin)
        poisson_layout.addLayout(scale_layout)
        
        # Density threshold
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("密度阈值 (0.0-0.5):"))
        density_spin = QDoubleSpinBox()
        density_spin.setRange(0.0, 0.5)
        density_spin.setValue(0.15)
        density_spin.setSingleStep(0.05)
        density_spin.setToolTip("过滤低密度区域。越高过滤越多。推荐: 0.1-0.2")
        density_layout.addWidget(density_spin)
        poisson_layout.addLayout(density_layout)
        
        poisson_group.setLayout(poisson_layout)
        layout.addWidget(poisson_group)
        
        # 提示信息
        info_label = QLabel("💡 提示:\n• 如果mesh质量差，尝试增加重建深度(10-11)\n• 如果有太多杂点，增加密度阈值(0.2-0.3)\n• 如果表面不完整，减小密度阈值(0.05-0.1)")
        info_label.setStyleSheet("color: #5BA3D8; font-size: 10px; padding: 10px; background-color: #2B2B3B; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("生成Mesh")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("取消")
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            try:
                # 获取参数
                method_idx = method_combo.currentIndex()
                method_map = ['poisson', 'alpha_shape', 'ball_pivoting']
                method = method_map[method_idx]
                
                depth = depth_spin.value()
                scale = scale_spin.value()
                density_threshold = density_spin.value()
                
                self.log(f"开始生成Mesh: 方法={method}, depth={depth}, scale={scale}, density={density_threshold}", "info")
                
                # 清除旧的mesh缓存
                if hasattr(self.edit_3d_viewer, 'clear_mesh_cache'):
                    self.edit_3d_viewer.clear_mesh_cache()
                
                # 获取当前可见点数据
                if not hasattr(self.edit_3d_viewer, 'get_render_data'):
                    self.show_message("错误", "3D查看器不支持mesh生成", "critical")
                    return
                
                positions, colors, indices = self.edit_3d_viewer.get_render_data()
                
                if positions is None or len(positions) == 0:
                    self.show_message("错误", "没有可用的点云数据", "critical")
                    return
                
                self.log(f"开始生成mesh，使用 {len(positions)} 个点...", "info")
                
                # 调用mesh生成方法
                if hasattr(self.edit_3d_viewer, 'generate_mesh_from_points'):
                    mesh_data = self.edit_3d_viewer.generate_mesh_from_points(
                        positions, colors,
                        method=method,
                        depth=depth,
                        scale=scale,
                        density_threshold=density_threshold
                    )
                    
                    if mesh_data is not None:
                        # 保存生成的mesh到缓存
                        self.edit_3d_viewer.cached_mesh = mesh_data
                        
                        # 启用mesh模式
                        if hasattr(self.edit_3d_viewer, 'enable_mesh_mode'):
                            self.edit_3d_viewer.enable_mesh_mode()
                        
                        # 切换到MESH渲染模式
                        from gaussian_multi_mode_renderer import RenderMode
                        if hasattr(self.edit_3d_viewer, 'set_render_mode'):
                            self.edit_3d_viewer.set_render_mode(RenderMode.MESH)
                            self.log("Mesh生成成功，已切换到Mesh模式", "info")
                            self.show_message("成功", "Mesh生成成功！", "info")
                        else:
                            self.show_message("错误", "3D查看器不支持Mesh模式", "critical")
                    else:
                        self.show_message("错误", "Mesh生成失败，请查看日志", "critical")
                else:
                    self.show_message("错误", "3D查看器不支持mesh生成", "critical")
                    
            except Exception as e:
                import traceback
                self.log(f"生成Mesh失败: {e}\n{traceback.format_exc()}", "error")
                self.show_message("错误", f"生成Mesh失败: {str(e)}", "critical")
    
    def export_mesh(self):
        """导出Mesh到文件"""
        if not MULTI_MODE_AVAILABLE:
            self.show_message("错误", "多模式渲染器不可用", "critical")
            return
        
        if not hasattr(self, 'edit_3d_viewer') or self.edit_3d_viewer is None:
            self.show_message("提示", "请先加载模型并打开3D视图", "warning")
            return
        
        if not hasattr(self.edit_3d_viewer, 'cached_mesh') or self.edit_3d_viewer.cached_mesh is None:
            self.show_message("提示", "请先生成Mesh（切换到Mesh模式）", "warning")
            return
        
        # 选择保存路径
        default_path = "mesh_export.ply"
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "导出Mesh", 
            default_path, 
            "PLY Files (*.ply);;OBJ Files (*.obj);;STL Files (*.stl)"
        )
        
        if path:
            try:
                # 根据文件扩展名确定格式
                ext = os.path.splitext(path)[1].lower()
                format_map = {'.ply': 'ply', '.obj': 'obj', '.stl': 'stl'}
                file_format = format_map.get(ext, 'ply')
                
                # 导出mesh
                if hasattr(self.edit_3d_viewer, 'export_mesh'):
                    success = self.edit_3d_viewer.export_mesh(path, format=file_format)
                    if success:
                        self.log(f"Mesh已导出到: {path}", "info")
                        self.show_message("成功", f"Mesh已导出到: {path}", "info")
                    else:
                        self.show_message("错误", "Mesh导出失败", "critical")
                else:
                    self.show_message("错误", "3D查看器不支持Mesh导出", "critical")
            except Exception as e:
                import traceback
                self.log(f"导出Mesh失败: {e}\n{traceback.format_exc()}", "error")
                self.show_message("错误", f"导出Mesh失败: {str(e)}", "critical")
    
    def clear_mesh_cache(self):
        """清除Mesh缓存"""
        if not MULTI_MODE_AVAILABLE:
            self.show_message("错误", "多模式渲染器不可用", "critical")
            return
        
        if not hasattr(self, 'edit_3d_viewer') or self.edit_3d_viewer is None:
            self.show_message("提示", "请先加载模型并打开3D视图", "warning")
            return
        
        try:
            if hasattr(self.edit_3d_viewer, 'clear_mesh_cache'):
                self.edit_3d_viewer.clear_mesh_cache()
                self.log("Mesh缓存已清除", "info")
                self.show_message("成功", "Mesh缓存已清除", "info")
            else:
                self.show_message("错误", "3D查看器不支持清除Mesh缓存", "critical")
        except Exception as e:
            import traceback
            self.log(f"清除Mesh缓存失败: {e}\n{traceback.format_exc()}", "error")
            self.show_message("错误", f"清除Mesh缓存失败: {str(e)}", "critical")
    
    def update_status(self, message, progress=None):
        """更新状态栏（线程安全）"""
        # 使用信号确保在主线程中更新UI
        self.status_signal.emit(message, progress)
    
    def _update_status_internal(self, message, progress=None):
        """内部状态更新方法，在主线程中调用"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(int(progress))
        else:
            self.progress_bar.setVisible(False)
    
    def update_system_info(self):
        """更新系统信息（线程安全）"""
        if torch.cuda.is_available():
            info = MemoryManager.get_gpu_memory_info()
        else:
            info = "CPU模式"
        # 使用信号确保在主线程中更新UI
        self.system_info_signal.emit(info)
    
    def _update_system_info_internal(self, info):
        """内部系统信息更新方法，在主线程中调用"""
        self.system_info_label.setText(info)
    
    def show_message(self, title, message, msg_type="info"):
        """显示消息框（线程安全）"""
        # 使用信号确保在主线程中显示
        self.show_message_signal.emit(title, message, msg_type)
    
    def _show_message_internal(self, title, message, msg_type="info"):
        """内部消息框方法，在主线程中调用"""
        if msg_type == "critical":
            QMessageBox.critical(self, title, message)
        elif msg_type == "warning":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.information(self, title, message)
    
    def _set_text_internal(self, widget, text):
        """内部设置文本方法，在主线程中调用"""
        widget.setText(text)
    
    def _set_enabled_internal(self, widget, enabled):
        """内部设置启用状态方法，在主线程中调用"""
        widget.setEnabled(enabled)
    
    def _set_colmap_status_internal(self, status):
        """内部设置COLMAP状态方法，在主线程中调用"""
        self.colmap_status_label.setText(status)
    
    def _show_question_dialog(self, title, message, callback_name):
        """显示问题对话框（在主线程中调用）"""
        reply = QMessageBox.question(self, title, message, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes and callback_name:
            # 使用QTimer延迟调用回调，确保在消息循环中执行
            QTimer.singleShot(0, lambda: getattr(self, callback_name, lambda: None)())
    
    def _update_editor_view_internal(self):
        """更新编辑器视图（在主线程中调用）"""
        if not EDITOR_AVAILABLE:
            return
        
        try:
            if hasattr(self, 'edit_3d_viewer') and hasattr(self, 'gaussian_editor') and self.gaussian_editor is not None:
                self.edit_3d_viewer.set_gaussians(self.gaussians, self.gaussian_editor)
                # 更新相机参数（使用第一个视图的相机）
                if hasattr(self, 'cameras') and len(self.cameras) > 0:
                    self.update_viewer_camera_params()
        except Exception as e:
            self.log(f"更新编辑器视图失败: {str(e)}", "warning")
    
    def _display_view_internal(self):
        """显示当前视图（在主线程中调用）"""
        self.display_current_view()
    
    def show_shortcuts(self):
        """显示快捷键帮助"""
        shortcuts = """
快捷键帮助:
F1 - 显示此帮助
Ctrl+R - 重置系统到初始状态
Ctrl+Q - 退出程序
"""
        QMessageBox.information(self, "快捷键", shortcuts)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """
GSSE - Gaussian Splatting Semantic Editor
高斯点云语义编辑器

版本: 1.0
基于PyQt5开发
支持多种分割方法（包括SAGS、SAGA等）
"""
        QMessageBox.about(self, "关于GSSE", about_text)
    
    def reset_system(self):
        """重置系统到初始状态"""
        # 询问用户确认
        reply = QMessageBox.question(
            self, 
            '确认重置', 
            '是否要重置系统到初始状态？\n\n这将:\n- 停止所有正在运行的进程\n- 卸载所有已加载的模型\n- 清除所有分割数据\n- 释放GPU内存\n\n注意: 未保存的分割结果将丢失！',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.log("=" * 50, "info")
        self.log("开始重置系统...", "info")
        
        try:
            # 1. 停止所有正在运行的进程
            self.log("正在停止所有进程...", "info")
            
            # 停止HTTP服务器
            self.stop_http_server()
            
            # 停止COLMAP处理
            if self.is_colmap_processing and self.colmap_processor:
                try:
                    self.colmap_processor.stop_processing()
                    self.log("已停止COLMAP处理", "info")
                except Exception as e:
                    self.log(f"停止COLMAP处理时出错: {str(e)}", "warning")
                self.is_colmap_processing = False
                self.colmap_processor = None
            
            # 停止训练进程
            if self.is_training and self.training_process:
                try:
                    self.training_process.terminate()
                    try:
                        self.training_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.training_process.kill()
                        self.training_process.wait()
                    self.log("已停止训练进程", "info")
                except Exception as e:
                    self.log(f"停止训练进程时出错: {str(e)}", "warning")
                self.is_training = False
                self.training_process = None
                self.training_thread = None
            
            # 停止SOG训练进程
            if self.is_sog_training and self.sog_process:
                try:
                    self.sog_process.terminate()
                    try:
                        self.sog_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.sog_process.kill()
                        self.sog_process.wait()
                    self.log("已停止SOG训练进程", "info")
                except Exception as e:
                    self.log(f"停止SOG训练进程时出错: {str(e)}", "warning")
                self.is_sog_training = False
                self.sog_process = None
                self.sog_thread = None
                self.sog_trainer = None
            
            # 停止SIBR查看器
            if self.sibr_process:
                try:
                    self.sibr_process.terminate()
                    self.sibr_process.wait(timeout=5)
                    self.log("已停止3D查看器", "info")
                except Exception as e:
                    self.log(f"停止3D查看器时出错: {str(e)}", "warning")
                self.sibr_process = None
            
            # 清理临时PLY文件
            if hasattr(self, 'temp_ply_path') and self.temp_ply_path:
                try:
                    temp_dir = os.path.dirname(self.temp_ply_path)
                    backup_ply = os.path.join(temp_dir, "point_cloud_original.ply")
                    if os.path.exists(backup_ply):
                        if os.path.exists(self.temp_ply_path):
                            os.remove(self.temp_ply_path)
                        os.rename(backup_ply, os.path.join(temp_dir, "point_cloud.ply"))
                    self.temp_ply_path = None
                    self.log("已清理临时文件", "info")
                except Exception as e:
                    self.log(f"清理临时文件时出错: {str(e)}", "warning")
            
            # 2. 清理模型和预测器
            self.log("正在卸载模型...", "info")
            
            # 清理Gaussians模型
            if self.gaussians is not None:
                del self.gaussians
                self.gaussians = None
            
            if self.scene is not None:
                del self.scene
                self.scene = None
            
            if self.cameras is not None:
                del self.cameras
                self.cameras = None
            
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            if self.background is not None:
                del self.background
                self.background = None
            
            # 清理SAM预测器
            if self.predictor is not None:
                del self.predictor
                self.predictor = None
            
            if self.preprocess_predictor is not None:
                del self.preprocess_predictor
                self.preprocess_predictor = None
            
            # 清理SAGA模块
            if self.saga_module is not None:
                del self.saga_module
                self.saga_module = None
            
            self.log("已卸载所有模型", "info")
            
            # 3. 重置所有状态变量
            self.log("正在重置状态变量...", "info")
            
            # 模型状态
            self.model_loaded = False
            self.sam_features = {}
            self.current_view_idx = 0
            self.current_image = None
            self.current_mask = None
            self.click_points = []
            self.current_point_type = 1
            
            # 预处理状态
            self.preprocess_points = []
            self.preprocess_current_point_type = 1
            self.preprocess_sam_features = {}
            self.preprocess_mask_id = 2
            self.original_images = {}
            
            # SAGA状态
            self.saga_feature_loaded = False
            self.saga_point_prompts = []
            self.saga_current_point_type = 1
            
            self.log("已重置状态变量", "info")
            
            # 4. 清理GPU内存
            self.log("正在清理GPU内存...", "info")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            self.log("已清理GPU内存", "info")
            
            # 5. 清空显示界面
            self.log("正在清空显示界面...", "info")
            
            # 清空画布
            if hasattr(self, 'original_canvas'):
                self.original_canvas.set_image(None)
                self.original_canvas.clear_points()
            if hasattr(self, 'segmentation_canvas'):
                self.segmentation_canvas.set_image(None)
                self.segmentation_canvas.clear_points()
            if hasattr(self, 'saga_canvas') and SAGA_AVAILABLE:
                self.saga_canvas.set_image(None)
                self.saga_canvas.clear_points()
            
            self.log("已清空显示界面", "info")
            
            # 6. 重置UI控件状态
            self.log("正在重置UI控件...", "info")
            
            # 重置按钮状态
            self.start_colmap_btn.setEnabled(True)
            self.stop_colmap_btn.setEnabled(False)
            self.start_training_btn.setEnabled(True)
            self.stop_training_btn.setEnabled(False)
            self.start_sog_training_btn.setEnabled(True)
            self.stop_sog_training_btn.setEnabled(False)
            
            # 重置状态标签
            self.colmap_status_label.setText("未开始COLMAP处理")
            self.training_status_label.setText("未开始训练")
            self.sog_training_status_label.setText("未开始SOG训练")
            self.render_status_label.setText("未加载模型")
            
            # 重置进度条
            if hasattr(self, 'colmap_progress_bar'):
                self.colmap_progress_bar.setValue(0)
            if hasattr(self, 'training_progress_bar'):
                self.training_progress_bar.setValue(0)
            if hasattr(self, 'sog_progress_bar'):
                self.sog_progress_bar.setValue(0)
            
            self.log("已重置UI控件", "info")
            
            # 7. 更新系统信息
            self.update_system_info()
            
            self.log("=" * 50, "info")
            self.log("系统重置完成！可以开始处理新的数据", "success")
            
            QMessageBox.information(
                self,
                '重置完成',
                '系统已成功重置到初始状态！\n\n现在可以处理新的数据了。',
                QMessageBox.Ok
            )
            
        except Exception as e:
            error_msg = f"重置系统时发生错误: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg, "error")
            QMessageBox.critical(
                self,
                '重置失败',
                f'重置系统时发生错误:\n{str(e)}',
                QMessageBox.Ok
            )
    
    def start_http_server(self):
        """启动HTTP服务器"""
        if not HTTP_SERVER_AVAILABLE:
            self.log("HTTP服务器模块不可用，跳过启动", "warning")
            return
        
        if self.http_server_started:
            self.log("HTTP服务器已在运行", "info")
            return
        
        try:
            self.log("正在启动HTTP服务器...", "info")
            self.http_server = get_global_server()
            
            if self.http_server.start():
                self.http_server_started = True
                port = self.http_server.get_port()
                base_url = self.http_server.get_base_url()
                self.log(f"HTTP服务器启动成功: {base_url}", "info")
                self.log("HTTP服务器将用于Cesium文件传输，解决file://协议限制", "info")
            else:
                self.log("HTTP服务器启动失败", "error")
                
        except Exception as e:
            self.log(f"启动HTTP服务器时出错: {str(e)}", "error")
    
    def stop_http_server(self):
        """停止HTTP服务器"""
        if not HTTP_SERVER_AVAILABLE or not self.http_server_started:
            return
        
        try:
            self.log("正在停止HTTP服务器...", "info")
            if self.http_server:
                self.http_server.stop()
            
            # 也停止全局服务器
            stop_global_server()
            
            self.http_server_started = False
            self.http_server = None
            self.log("HTTP服务器已停止", "info")
            
        except Exception as e:
            self.log(f"停止HTTP服务器时出错: {str(e)}", "error")
    
    def clear_gpu_memory(self):
        """清理GPU内存"""
        MemoryManager.clear_gpu_memory()
        self.log("GPU内存已清理", "info")
        self.update_system_info()
    
    def show_memory_status(self):
        """显示内存状态"""
        info = MemoryManager.get_gpu_memory_info()
        QMessageBox.information(self, "内存状态", info)
    
    # 文件浏览方法
    def browse_model_path(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录", self.model_path_edit.text())
        if path:
            self.model_path_edit.setText(path)
    
    def browse_source_path(self):
        """浏览数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择数据目录", self.source_path_edit.text())
        if path:
            self.source_path_edit.setText(path)
    
    def saga_browse_model_path(self):
        """浏览SAGA模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录", self.saga_model_path_edit.text())
        if path:
            self.saga_model_path_edit.setText(path)
    
    def saga_browse_source_path(self):
        """浏览SAGA数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择数据目录", self.saga_source_path_edit.text())
        if path:
            self.saga_source_path_edit.setText(path)
    
    def browse_training_data_path(self):
        """浏览训练数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择训练数据目录", self.training_data_path_edit.text())
        if path:
            self.training_data_path_edit.setText(path)
            self.update_training_output_path(path)
    
    def browse_training_output_path(self):
        """浏览训练输出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.training_output_path_edit.text())
        if path:
            self.training_output_path_edit.setText(path)
    
    def update_training_output_path(self, training_data_path):
        """根据训练数据路径自动更新输出路径"""
        if training_data_path:
            sags_dir = os.path.dirname(__file__)
            folder_name = os.path.basename(training_data_path.rstrip('/'))
            output_path = os.path.join(sags_dir, "gaussiansplatting", "output", folder_name)
            self.training_output_path_edit.setText(output_path)
    
    # 模型加载相关方法
    def load_model(self):
        """加载3DGS模型"""
        def load_thread():
            try:
                self.update_status("正在加载模型...", 0)
                self.log("开始加载模型...")
                
                model_path = self.model_path_edit.text()
                source_path = self.source_path_edit.text()
                iteration = int(self.iteration_edit.text())
                resolution = int(self.resolution_combo.currentText())
                
                if not model_path:
                    self.show_message("错误", "请选择模型路径", "critical")
                    return
                
                # 若未填写数据路径，尝试从 cfg_args 读取
                images_dir = 'images'
                white_background = False
                if not source_path:
                    cfg_path = os.path.join(model_path, 'cfg_args')
                    if os.path.isfile(cfg_path):
                        try:
                            with open(cfg_path, 'r') as f:
                                cfg_ns = eval(f.read().strip())
                            source_path = getattr(cfg_ns, 'source_path', '')
                            images_dir = getattr(cfg_ns, 'images', 'images')
                            white_background = getattr(cfg_ns, 'white_background', False)
                            self.set_text_signal.emit(self.source_path_edit, source_path)
                        except Exception:
                            pass
                
                # 校验数据路径
                if not source_path or not (os.path.exists(os.path.join(source_path, 'sparse')) or os.path.exists(os.path.join(source_path, 'transforms_train.json'))):
                    self.show_message("错误", "无法识别数据集类型", "critical")
                    return
                
                # 构建scene_args
                import argparse
                scene_args = argparse.Namespace(
                    model_path=model_path,
                    source_path=source_path,
                    images=images_dir,
                    white_background=white_background,
                    eval=False,
                    sh_degree=3,
                    resolution=resolution,
                    data_device='cuda',
                )
                
                # 初始化高斯模型和场景
                self.gaussians = GaussianModel(scene_args.sh_degree)
                self.scene = Scene(scene_args, self.gaussians, load_iteration=iteration, shuffle=False)
                self.cameras = self.scene.getTrainCameras()
                
                # 设置渲染参数
                bg_color = [1, 1, 1] if scene_args.white_background else [0, 0, 0]
                self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
                
                # 设置pipeline
                self.pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
                
                # 设置source_path属性，用于自动地理定位
                self.gaussians.source_path = source_path
                
                # 更新视图控制
                self.view_spinbox.setMaximum(max(0, len(self.cameras)-1))
                self.view_spinbox.setValue(0)
                
                # 初始化SAGA模块（如果可用）
                if SAGA_AVAILABLE and self.saga_module is None:
                    try:
                        self.saga_module = SAGAModule(model_path, source_path)
                        if self.saga_module.load_model(iteration=iteration):
                            self.log("SAGA模块初始化成功")
                        else:
                            self.log("SAGA模块初始化失败", "warning")
                    except Exception as e:
                        self.log(f"SAGA模块初始化错误: {str(e)}", "warning")
                
                # 初始化编辑器（如果可用）
                if EDITOR_AVAILABLE:
                    try:
                        self.gaussian_editor = GaussianEditor(self.gaussians)
                        self.log("3DGS编辑器初始化成功")
                        # 通过信号更新3D编辑视图（必须在主线程中执行）
                        self.update_editor_view_signal.emit()
                    except Exception as e:
                        self.log(f"编辑器初始化错误: {str(e)}", "warning")
                
                self.model_loaded = True
                self.update_status("模型加载完成", 100)
                self.log(f"模型加载完成，共{len(self.cameras)}个视图")
                
                # 启用GIS功能
                if GIS_AVAILABLE:
                    self.gis_export_btn.setEnabled(True)
                    self.gis_load_to_cesium_btn.setEnabled(True)
                
                # 显示第一个视图（通过信号在主线程中执行）
                self.current_view_idx = 0
                self.display_view_signal.emit()
                
                # 更新系统信息
                self.update_system_info()
                
            except Exception as e:
                self.update_status("模型加载失败", 0)
                self.log(f"加载模型失败: {str(e)}", "error")
                self.show_message("错误", f"加载模型失败: {str(e)}", "critical")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def saga_load_model(self):
        """加载3DGS模型（SAGA专用）"""
        def load_thread():
            try:
                self.update_status("正在加载模型...", 0)
                self.log("开始加载模型（SAGA）...")
                
                model_path = self.saga_model_path_edit.text()
                source_path = self.saga_source_path_edit.text()
                iteration = int(self.saga_iteration_edit.text())
                resolution = int(self.saga_resolution_combo.currentText())
                
                if not model_path:
                    self.show_message("错误", "请选择模型路径", "critical")
                    return
                
                # 若未填写数据路径，尝试从 cfg_args 读取
                images_dir = 'images'
                white_background = False
                if not source_path:
                    cfg_path = os.path.join(model_path, 'cfg_args')
                    if os.path.isfile(cfg_path):
                        try:
                            with open(cfg_path, 'r') as f:
                                cfg_ns = eval(f.read().strip())
                            source_path = getattr(cfg_ns, 'source_path', '')
                            images_dir = getattr(cfg_ns, 'images', 'images')
                            white_background = getattr(cfg_ns, 'white_background', False)
                            self.set_text_signal.emit(self.saga_source_path_edit, source_path)
                        except Exception:
                            pass
                
                # 校验数据路径
                if not source_path or not (os.path.exists(os.path.join(source_path, 'sparse')) or os.path.exists(os.path.join(source_path, 'transforms_train.json'))):
                    self.show_message("错误", "无法识别数据集类型", "critical")
                    return
                
                # 构建scene_args
                import argparse
                scene_args = argparse.Namespace(
                    model_path=model_path,
                    source_path=source_path,
                    images=images_dir,
                    white_background=white_background,
                    eval=False,
                    sh_degree=3,
                    resolution=resolution,
                    data_device='cuda',
                )
                
                # 初始化高斯模型和场景
                self.gaussians = GaussianModel(scene_args.sh_degree)
                self.scene = Scene(scene_args, self.gaussians, load_iteration=iteration, shuffle=False)
                self.cameras = self.scene.getTrainCameras()
                
                # 设置渲染参数
                bg_color = [1, 1, 1] if scene_args.white_background else [0, 0, 0]
                self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
                
                # 设置pipeline
                self.pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
                
                # 设置source_path属性，用于自动地理定位
                self.gaussians.source_path = source_path
                
                # 更新视图控制（SAGA使用相同的视图控制）
                self.view_spinbox.setMaximum(max(0, len(self.cameras)-1))
                self.view_spinbox.setValue(0)
                
                # 初始化SAGA模块（如果可用）
                if SAGA_AVAILABLE and self.saga_module is None:
                    try:
                        self.saga_module = SAGAModule(model_path, source_path)
                        if self.saga_module.load_model(iteration=iteration):
                            self.log("SAGA模块初始化成功")
                        else:
                            self.log("SAGA模块初始化失败", "warning")
                    except Exception as e:
                        self.log(f"SAGA模块初始化错误: {str(e)}", "warning")
                
                self.model_loaded = True
                self.update_status("模型加载完成", 100)
                self.log(f"模型加载完成（SAGA），共{len(self.cameras)}个视图")
                
                # 启用GIS功能
                if GIS_AVAILABLE:
                    self.gis_export_btn.setEnabled(True)
                    self.gis_load_to_cesium_btn.setEnabled(True)
                
                # 显示第一个视图（通过信号在主线程中执行）
                self.current_view_idx = 0
                self.display_view_signal.emit()
                
                # 更新系统信息
                self.update_system_info()
                
            except Exception as e:
                self.update_status("模型加载失败", 0)
                self.log(f"加载模型失败（SAGA）: {str(e)}", "error")
                self.show_message("错误", f"加载模型失败: {str(e)}", "critical")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_model_sync(self):
        """同步加载3DGS模型（用于一键式流程）"""
        try:
            self.update_status("正在加载模型...", 0)
            self.log("开始加载模型...")
            
            # 构建参数
            model_path = self.model_path_edit.text()
            source_path = self.source_path_edit.text()
            iteration = int(self.iteration_edit.text())
            resolution = int(self.resolution_combo.currentText())
            
            if not model_path:
                self.log("模型路径为空", "error")
                return False

            # 若未填写数据路径，尝试从 cfg_args 读取
            images_dir = 'images'
            white_background = False
            if not source_path:
                cfg_path = os.path.join(model_path, 'cfg_args')
                if os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, 'r') as f:
                            cfg_ns = eval(f.read().strip())
                        source_path = getattr(cfg_ns, 'source_path', '')
                        images_dir = getattr(cfg_ns, 'images', 'images')
                        white_background = getattr(cfg_ns, 'white_background', False)
                        self.set_text_signal.emit(self.source_path_edit, source_path)
                    except Exception:
                        pass

            # 校验数据路径：需包含 COLMAP 的 sparse 目录，或 Blender 的 transforms_train.json
            if not source_path or not (os.path.exists(os.path.join(source_path, 'sparse')) or os.path.exists(os.path.join(source_path, 'transforms_train.json'))):
                self.log("无法识别数据集类型", "error")
                return False
            
            # 构建scene_args
            import argparse
            scene_args = argparse.Namespace(
                model_path=model_path,
                source_path=source_path,
                images=images_dir,
                white_background=white_background,
                eval=False,
                sh_degree=3,
                resolution=resolution,
                data_device='cuda',
            )
            
            # 初始化高斯模型和场景
            self.gaussians = GaussianModel(scene_args.sh_degree)
            self.scene = Scene(scene_args, self.gaussians, load_iteration=iteration, shuffle=False)
            self.cameras = self.scene.getTrainCameras()
            
            # 设置渲染参数
            bg_color = [1, 1, 1] if scene_args.white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
            
            # 设置pipeline
            self.pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            
            # 更新视图控制
            self.view_spinbox.setMaximum(max(0, len(self.cameras)-1))
            self.view_spinbox.setValue(0)
            
            self.model_loaded = True
            self.update_status("模型加载完成", 100)
            self.log(f"模型加载完成，共{len(self.cameras)}个视图")
            
            # 启用GIS功能
            if GIS_AVAILABLE:
                self.gis_export_btn.setEnabled(True)
                self.gis_load_to_cesium_btn.setEnabled(True)
            
            # 显示第一个视图
            self.current_view_idx = 0
            self.display_current_view()
            
            # 更新系统信息
            self.update_system_info()
            
            return True
            
        except Exception as e:
            self.update_status("模型加载失败", 0)
            self.log(f"加载模型失败: {str(e)}", "error")
            traceback.print_exc()
            return False
    
    def get_available_iterations(self, model_path):
        """获取可用的迭代次数列表"""
        point_cloud_dir = os.path.join(model_path, "point_cloud")
        if not os.path.exists(point_cloud_dir):
            return []
        
        iterations = []
        for item in os.listdir(point_cloud_dir):
            if item.startswith("iteration_") and os.path.isdir(os.path.join(point_cloud_dir, item)):
                try:
                    iteration_num = int(item.split("_")[1])
                    iterations.append(iteration_num)
                except (ValueError, IndexError):
                    continue
        
        return sorted(iterations)
    
    def select_iteration_dialog(self, model_path):
        """显示迭代次数选择对话框"""
        available_iterations = self.get_available_iterations(model_path)
        
        if not available_iterations:
            self.show_message("错误", "未找到任何可用的迭代结果", "critical")
            return None
        
        # 创建自定义对话框
        dialog = QMessageBox(self)
        dialog.setWindowTitle("选择模型")
        dialog.setText("请选择要查看的模型迭代次数:")
        dialog.setIcon(QMessageBox.Question)
        
        # 创建组合框用于选择
        combo = QComboBox()
        for iteration in available_iterations:
            display_text = f"iteration_{iteration}"
            # 如果有分割结果且当前迭代是分割时的迭代，添加标注
            if hasattr(self, 'final_mask') and hasattr(self, 'segmentation_iteration') and iteration == self.segmentation_iteration:
                display_text += " （本次分割）"
            combo.addItem(display_text, iteration)
        
        # 创建布局添加组合框
        layout = dialog.layout()
        layout.addWidget(combo, 1, 1, 1, layout.columnCount())
        
        # 添加按钮
        ok_button = dialog.addButton("确定", QMessageBox.AcceptRole)
        cancel_button = dialog.addButton("取消", QMessageBox.RejectRole)
        
        # 显示对话框
        if dialog.exec_() == QMessageBox.AcceptRole:
            selected_iteration = combo.currentData()
            return selected_iteration
        
        return None
    
    def multi_point_sam_prompt(self, predictor, image, input_points, input_labels, mask_id):
        """使用多个点提示进行SAM分割"""
        try:
            # Check if this is a FastSAM model
            if hasattr(predictor, 'model') and hasattr(predictor.model, 'names'):
                # This is a FastSAM model
                input_points_tensor = torch.tensor(input_points, dtype=torch.float32)
                return fastsam_point_prompt(predictor, image, input_points_tensor, mask_id)
            else:
                # Regular SAM model
                # 设置图像
                predictor.set_image(image)
                
                # 使用多个点提示进行预测
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
                
                # 选择最佳掩码（通常是得分最高的）
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                
                # 根据mask_id选择掩码
                if mask_id < len(masks):
                    selected_mask = masks[mask_id]
                else:
                    selected_mask = best_mask
                
                return selected_mask.astype(np.uint8)
            
        except Exception as e:
            self.log(f"多点SAM分割失败: {str(e)}", "error")
            # 回退到单点分割
            if len(input_points) > 0:
                single_point = input_points[0:1]
                single_label = input_labels[0:1]
                masks, scores, logits = predictor.predict(
                    point_coords=single_point,
                    point_labels=single_label,
                    multimask_output=True,
                )
                best_mask_idx = np.argmax(scores)
                return masks[best_mask_idx].astype(np.uint8)
            else:
                raise e
    
    def display_current_view(self):
        """显示当前视图"""
        if not self.model_loaded or not self.cameras:
            return
        
        try:
            view = self.cameras[self.current_view_idx]
            
            # 如果编辑器存在，临时设置已删除点的opacity为0
            original_opacity = None
            if EDITOR_AVAILABLE and self.gaussian_editor:
                visible_indices = self.gaussian_editor.get_visible_indices()
                # 确保visible_indices在正确的设备上
                device = self.gaussians.get_xyz.device
                if visible_indices.device != device:
                    visible_indices = visible_indices.to(device)
                
                if len(visible_indices) < self.gaussians.get_xyz.shape[0]:
                    # 有被删除的点，需要过滤
                    # 保存原始opacity
                    original_opacity = self.gaussians._opacity.clone()
                    # 设置已删除点的opacity为极小值（接近0）
                    # 使用data属性避免梯度问题
                    all_indices = torch.arange(self.gaussians.get_xyz.shape[0], device=device)
                    
                    # 使用更可靠的方法检查哪些点被删除
                    # 创建一个mask，标记所有可见的点
                    visible_mask = torch.zeros(self.gaussians.get_xyz.shape[0], dtype=torch.bool, device=device)
                    visible_mask[visible_indices] = True
                    deleted_mask = ~visible_mask
                    
                    if deleted_mask.any():
                        with torch.no_grad():
                            self.gaussians._opacity.data[deleted_mask] = -10.0  # sigmoid(-10) ≈ 0
            
            # 渲染图像
            with torch.no_grad():
                render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
            
            # 恢复原始opacity（如果被修改）
            if original_opacity is not None:
                with torch.no_grad():
                    self.gaussians._opacity.data.copy_(original_opacity)
            
            # 转换为uint8格式
            render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
            self.original_canvas.set_image(render_image)
            self.current_image = render_image
            
            self.log(f"显示视图 {self.current_view_idx}: {view.image_name}")
            
            # 如果有分割结果，也更新分割结果显示
            if hasattr(self, 'final_mask'):
                self.display_segmentation_result()
            
            # 如果SAGA可用且已加载，更新SAGA可视化
            # 移除标签页限制，确保SAGA可视化始终更新
            if SAGA_AVAILABLE and self.saga_module and self.model_loaded:
                if hasattr(self, 'saga_canvas'):
                    self.saga_update_visualization()
            
        except Exception as e:
            self.log(f"显示视图失败: {str(e)}", "error")
            traceback.print_exc()
    
    def on_display_tab_changed(self, index):
        """标签页切换事件处理"""
        if not SAGA_AVAILABLE:
            return
        
        # 检查是否切换到SAGA标签页（需要根据实际标签页顺序确定索引）
        # 标签页顺序：3D渲染(0), SAGS分割(1), SAGA分割(2)
        if hasattr(self, 'saga_canvas'):
            # 找到SAGA标签页的索引
            saga_tab_index = -1
            for i in range(self.display_tabs.count()):
                if self.display_tabs.widget(i) == self.saga_canvas:
                    saga_tab_index = i
                    break
            
            # 如果切换到SAGA标签页，且模型已加载，更新可视化
            if index == saga_tab_index:
                if self.model_loaded and self.saga_module:
                    self.saga_update_visualization()
    
    # 视图控制方法
    def change_view(self):
        """改变视图"""
        idx = self.view_spinbox.value()
        if 0 <= idx < len(self.cameras) if self.cameras else False:
            self.current_view_idx = idx
            self.display_current_view()
    
    def prev_view(self):
        """上一视图"""
        if self.cameras and self.current_view_idx > 0:
            self.current_view_idx -= 1
            self.view_spinbox.setValue(self.current_view_idx)
    
    def next_view(self):
        """下一视图"""
        if self.cameras and self.current_view_idx < len(self.cameras) - 1:
            self.current_view_idx += 1
            self.view_spinbox.setValue(self.current_view_idx)
    
    # 提示方式相关方法
    def on_prompt_type_change(self):
        """提示方式改变"""
        is_point = self.point_prompt_radio.isChecked()
        self.text_prompt_edit.setEnabled(not is_point)
    
    # 点提示管理方法
    def clear_all_points(self):
        """清除所有点"""
        self.original_canvas.clear_points()
        self.click_points = []
        self.update_points_display()
        self.log("已清除所有点提示", "info")
    
    def remove_last_point(self):
        """移除最后一个点"""
        if self.original_canvas.click_points:
            self.original_canvas.remove_last_point()
            self.click_points = self.original_canvas.click_points
            self.update_points_display()
            self.log("已移除最后一个点", "info")
    
    def update_points_display(self):
        """更新点列表显示"""
        self.points_listbox.clear()
        for i, (x, y, point_type) in enumerate(self.click_points):
            point_type_str = "正点" if point_type == 1 else "负点"
            self.points_listbox.addItem(f"{i+1}. {point_type_str} ({x:.0f}, {y:.0f})")
    
    # 分割相关方法
    def preprocess_sam(self):
        """预处理SAM特征"""
        if not self.model_loaded:
            self.show_message("警告", "请先加载模型", "warning")
            return
        
        def preprocess_thread():
            try:
                self.update_status("正在预处理SAM特征...", 0)
                self.log("开始预处理SAM特征...")
                
                sam_model_type = self.sam_model_combo.currentText()
                sam_long_side = int(self.sam_long_side_combo.currentText())
                
                # 获取SAM模型路径
                sam_model_paths = {
                    'vit_h': 'sam_vit_h_4b8939.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth', 
                    'vit_b': 'sam_vit_b_01ec64.pth',
                    'fastsam_s': 'FastSAM-s.pt',
                    'fastsam_x': 'FastSAM-x.pt'
                }
                
                sam_ckpt_filename = sam_model_paths.get(sam_model_type, 'sam_vit_b_01ec64.pth')
                sam_ckpt_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
                
                if not os.path.isfile(sam_ckpt_path):
                    self.log(f"SAM模型文件不存在: {sam_ckpt_path}", "error")
                    return
                
                # 初始化SAM预测器
                if sam_model_type.startswith('fastsam'):
                    self.predictor = init_fastsam_predictor(sam_ckpt_path, model_type=sam_model_type)
                else:
                    self.predictor = init_sam_predictor(sam_ckpt_path, model_type=sam_model_type)
                
                # 预处理每个视图的SAM特征
                self.preprocess_sam_features_for_sags(sam_long_side)
                
                self.update_status("SAM特征预处理完成", 100)
                self.log("SAM特征预处理完成")
                
            except Exception as e:
                self.update_status("预处理失败", 0)
                self.log(f"SAM特征预处理失败: {str(e)}", "error")
                self.show_message("错误", f"预处理失败: {str(e)}", "critical")
        
        threading.Thread(target=preprocess_thread, daemon=True).start()
    
    def preprocess_sam_features_for_sags(self, sam_long_side):
        """为SAGS分割预处理SAM特征"""
        try:
            self.sam_features = {}
            self.log('预处理: 提取SAM特征...')
            
            # Get the current SAM model type
            sam_model = self.sam_model_combo.currentText()
            
            for i, view in enumerate(self.cameras):
                image_name = view.image_name
                self.log(f"处理视图 {i+1}/{len(self.cameras)}: {image_name}")
                
                with torch.no_grad():
                    render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                    render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
                render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
                
                # 调整图像大小以减少内存使用
                h, w = render_image.shape[:2]
                long_side = max(h, w)
                if long_side > sam_long_side:
                    scale = sam_long_side / long_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    render_image = cv2.resize(render_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Handle FastSAM vs SAM differently
                if sam_model.startswith('fastsam'):
                    # FastSAM doesn't need feature extraction - store image for later use
                    self.sam_features[image_name] = render_image
                else:
                    try:
                        self.predictor.set_image(render_image)
                    except torch.cuda.OutOfMemoryError:
                        self.log("GPU内存不足，清理内存后重试...", "warning")
                        MemoryManager.clear_gpu_memory()
                        self.predictor.set_image(render_image)
                    
                    # 将特征移到CPU以释放GPU内存
                    feats = self.predictor.features
                    if isinstance(feats, torch.Tensor):
                        feats = feats.cpu()
                    self.sam_features[image_name] = feats
                    del feats
                
                # 定期清理内存
                if i % 3 == 0:
                    MemoryManager.clear_gpu_memory()
            
            self.log(f"SAM预处理完成，处理了 {len(self.sam_features)} 个视图")
            
        except Exception as e:
            self.log(f"SAM特征预处理失败: {str(e)}", "error")
    
    def run_sags_with_preprocess(self):
        """使用预处理参数执行SAGS分割"""
        try:
            if not self.model_loaded:
                self.log("请先加载模型", "error")
                return
            
            # 获取SAM参数
            sam_model = self.sam_model_combo.currentText()
            sam_long_side = int(self.sam_long_side_combo.currentText())
            
            # 初始化SAM预测器（如果还没有）
            if not self.predictor:
                self.log("正在初始化SAM预测器...")
                
                # 获取SAM模型路径
                sam_model_paths = {
                    'vit_h': 'sam_vit_h_4b8939.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth', 
                    'vit_b': 'sam_vit_b_01ec64.pth',
                    'fastsam_s': 'FastSAM-s.pt',
                    'fastsam_x': 'FastSAM-x.pt'
                }
                
                sam_ckpt_filename = sam_model_paths.get(sam_model, 'sam_vit_b_01ec64.pth')
                sam_ckpt_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
                
                if not os.path.isfile(sam_ckpt_path):
                    self.log(f"SAM模型文件不存在: {sam_ckpt_path}", "error")
                    return
                
                # 初始化SAM预测器
                if sam_model.startswith('fastsam'):
                    self.predictor = init_fastsam_predictor(sam_ckpt_path, model_type=sam_model)
                else:
                    self.predictor = init_sam_predictor(sam_ckpt_path, model_type=sam_model)
            
            # 预处理SAM特征（如果还没有）
            if not self.sam_features:
                self.log("正在预处理SAM特征...")
                self.preprocess_sam_features_for_sags(sam_long_side)
            
            self.log("开始执行SAGS分割...")
            
            # 获取预处理参数
            prompt_type = "point" if self.preprocess_point_prompt_radio.isChecked() else "text"
            mask_id = self.preprocess_mask_id
            
            if prompt_type == "point":
                if not self.preprocess_points:
                    self.log("请先添加预处理点提示", "error")
                    return
                
                # 使用预处理点提示
                positive_points = [(x, y) for x, y, label in self.preprocess_points if label == 1]
                negative_points = [(x, y) for x, y, label in self.preprocess_points if label == 0]
                
                if not positive_points:
                    self.log("请至少添加一个正点", "error")
                    return
                
                input_points = np.array(positive_points)
                input_labels = np.ones(len(positive_points))
                
                if negative_points:
                    neg_points = np.array(negative_points)
                    neg_labels = np.zeros(len(negative_points))
                    input_points = np.vstack([input_points, neg_points])
                    input_labels = np.concatenate([input_labels, neg_labels])
                
                self.log(f"使用预处理 {len(positive_points)} 个正点和 {len(negative_points)} 个负点进行分割")
                
                # 生成3D提示（使用第一个正点）
                first_view = self.cameras[0]
                xyz = self.gaussians.get_xyz
                prompts_3d = generate_3d_prompts(xyz, first_view, input_points[:1].tolist())
                
            else:  # text prompt
                text_prompt = self.preprocess_text_edit.text()
                if not text_prompt:
                    self.log("请输入预处理文本提示", "error")
                    return
                
                input_points = None
                input_labels = None
                prompts_3d = None
                self.log(f"使用预处理文本提示进行分割: {text_prompt}")
            
            # 多视角分割
            multiview_masks = []
            sam_masks = []
            
            # 获取FastSAM参数
            fastsam_conf = self.fastsam_conf_slider.value() / 100.0
            fastsam_iou = self.fastsam_iou_slider.value() / 100.0
            
            for i, view in enumerate(self.cameras):
                image_name = view.image_name
                self.log(f"处理视图 {i+1}/{len(self.cameras)}: {image_name}")
                
                with torch.no_grad():
                    render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                    render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
                render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
                
                try:
                    if prompt_type == "point":
                        if sam_model.startswith('fastsam'):
                            prompts_2d = porject_to_2d(view, prompts_3d)
                            prompts_2d_tensor = torch.tensor(prompts_2d, dtype=torch.float32)
                            sam_mask_np = fastsam_point_prompt(self.predictor, render_image, prompts_2d_tensor, mask_id, conf=fastsam_conf, iou=fastsam_iou)
                        else:
                            prompts_2d = porject_to_2d(view, prompts_3d)
                            # 确保特征在CUDA上
                            feats = self.sam_features[image_name]
                            pred_device = getattr(self.predictor, 'device', next(self.predictor.model.parameters()).device)
                            if isinstance(feats, torch.Tensor) and feats.device.type != pred_device.type:
                                feats = feats.to(pred_device, non_blocking=True)
                            
                            # 使用多个点提示进行SAM分割
                            sam_mask_np = self.multi_point_sam_prompt(self.predictor, render_image, input_points, input_labels, mask_id)
                    else:
                        if sam_model.startswith('fastsam'):
                            sam_mask_np = fastsam_text_prompt(self.predictor, render_image, text_prompt, mask_id, conf=fastsam_conf, iou=fastsam_iou)
                        else:
                            sam_mask_np = text_prompting(self.predictor, render_image, text_prompt, mask_id)
                    
                except Exception as e:
                    self.log(f"分割失败 {image_name}: {str(e)}", "warning")
                    continue
                
                sam_mask = torch.from_numpy(sam_mask_np).to(self.gaussians.get_xyz.device)
                if len(sam_mask.shape) != 2:
                    sam_mask = sam_mask.squeeze(-1)
                sam_mask = sam_mask.long()
                sam_masks.append(sam_mask)
                
                # 确保xyz和sam_mask在同一设备上
                xyz = self.gaussians.get_xyz
                if xyz.device.type == 'cpu':
                    xyz_cpu = xyz.cpu()
                    point_mask, indices_mask = mask_inverse(xyz_cpu, view, sam_mask)
                else:
                    point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)
                multiview_masks.append(point_mask.unsqueeze(-1))
            
            # 多视角投票
            threshold = self.threshold_slider.value() / 100.0
            if sam_model == 'vit_b':
                adjusted_threshold = max(0.3, threshold - 0.2)
            elif sam_model == 'vit_l':
                adjusted_threshold = max(0.4, threshold - 0.1)
            elif sam_model.startswith('fastsam'):
                adjusted_threshold = max(0.4, threshold - 0.1)
            else:
                adjusted_threshold = threshold
            
            _, final_mask = ensemble(multiview_masks, threshold=adjusted_threshold)
            
            # 保存结果到iteration_666文件夹
            base_model_path = self.model_path_edit.text()
            output_dir = os.path.join(base_model_path, "point_cloud", "iteration_666")
            os.makedirs(output_dir, exist_ok=True)
            
            save_path = os.path.join(output_dir, 'point_cloud.ply')
            save_gd_path = os.path.join(output_dir, 'point_cloud_gd.ply')
            
            # 保存分割结果
            self.log(f'保存分割结果: {save_path}')
            save_gs(self.gaussians, final_mask, save_path)
            
            # 可选的高斯分解
            gd_interval = int(self.gd_interval_combo.currentText())
            if gd_interval != -1:
                self.log('应用高斯分解...')
                for i, view in enumerate(self.cameras):
                    if i % gd_interval == 0:
                        input_mask = sam_masks[i]
                        self.gaussians = gaussian_decomp(self.gaussians, view, input_mask, final_mask.to(self.gaussians.get_xyz.device))
                self.log(f'保存高斯分解结果: {save_gd_path}')
                save_gs(self.gaussians, final_mask, save_gd_path)
            
            self.log(f"SAGS分割完成，选中 {len(final_mask)} 个高斯点")
            self.log(f"结果保存在: {output_dir}")
            
            # 【关键】在重新加载模型之前保存原始模型和分割mask
            # 因为重新加载iteration_666会加载分割后的小模型
            original_num_gaussians = self.gaussians.get_xyz.shape[0]
            self.log(f"保存原始模型用于对比（点数: {original_num_gaussians}）", "info")
            
            # 保存原始模型引用（在重新加载之前！）
            if not hasattr(self, 'original_gaussians') or self.original_gaussians is None:
                self.original_gaussians = self.gaussians
            
            # 保存分割mask供3D渲染使用
            self.final_mask = final_mask
            
            # 更新模型路径为分割结果（保持原始模型路径，只更新迭代次数）
            self.set_text_signal.emit(self.iteration_edit, "666")
            
            # 重新加载模型以显示分割结果
            # 注意：这会加载iteration_666（只包含分割后的点）
            self.load_model_sync()
            
            # 检查重新加载后的模型大小
            new_num_gaussians = self.gaussians.get_xyz.shape[0]
            self.log(f"重新加载后的模型大小: {new_num_gaussians}", "info")
            
            if new_num_gaussians != original_num_gaussians:
                self.log(f"提示: 已加载分割后的模型（{new_num_gaussians}个点），原始模型有{original_num_gaussians}个点", "info")
                self.log(f"对比功能将使用保存的原始模型", "info")
                
                # final_mask仍然基于原始模型的索引，这是正确的
                # 因为对比功能需要在original_gaussians上应用这个mask
            
            # 更新3D渲染器显示分割结果
            # 注意：此时self.original_gaussians是完整模型，self.gaussians是分割后的小模型
            self.update_3d_viewer_with_segmentation()
            
        except Exception as e:
            self.log(f"SAGS分割失败: {str(e)}", "error")
            traceback.print_exc()
    
    def run_segmentation(self):
        """执行分割"""
        if not self.model_loaded or not self.predictor:
            self.show_message("警告", "请先加载模型并预处理SAM特征", "warning")
            return
        
        def segmentation_thread():
            try:
                self.update_status("正在执行分割...", 0)
                self.log("开始执行分割...")
                
                threshold = self.threshold_slider.value() / 100.0
                mask_id = int(self.mask_id_combo.currentText())
                is_point_prompt = self.point_prompt_radio.isChecked()
                sam_model_type = self.sam_model_combo.currentText()
                sam_long_side = int(self.sam_long_side_combo.currentText())
                
                xyz = self.gaussians.get_xyz
                
                if is_point_prompt:
                    if not self.click_points:
                        self.show_message("警告", "请先在图像上点击选择分割点", "warning")
                        return
                    
                    positive_points = [(x, y) for x, y, label in self.click_points if label == 1]
                    negative_points = [(x, y) for x, y, label in self.click_points if label == 0]
                    
                    if not positive_points:
                        self.show_message("警告", "请至少添加一个正点", "warning")
                        return
                    
                    input_points = np.asarray(positive_points, dtype=np.int32)
                    input_labels = np.ones(len(positive_points), dtype=np.int32)
                    
                    if negative_points:
                        neg_points = np.asarray(negative_points, dtype=np.int32)
                        neg_labels = np.zeros(len(negative_points), dtype=np.int32)
                        input_points = np.vstack([input_points, neg_points])
                        input_labels = np.concatenate([input_labels, neg_labels])
                    
                    self.log(f"使用 {len(positive_points)} 个正点和 {len(negative_points)} 个负点进行分割")
                    
                    first_view = self.cameras[0]
                    prompts_3d = generate_3d_prompts(xyz, first_view, positive_points[:1])
                else:
                    text_prompt = self.text_prompt_edit.text()
                    if not text_prompt:
                        self.show_message("警告", "请输入文本提示", "warning")
                        return
                    input_points = None
                    input_labels = None
                    prompts_3d = None
                
                # 多视角分割
                multiview_masks = []
                sam_masks = []
                
                for i, view in enumerate(self.cameras):
                    progress = int((i / len(self.cameras)) * 80)
                    self.update_status(f"正在处理视图 {i+1}/{len(self.cameras)}...", progress)
                    self.log(f"处理视图 {i+1}/{len(self.cameras)}: {view.image_name}")
                    
                    with torch.no_grad():
                        render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
                    
                    render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
                    
                    # 调整大小
                    h, w = render_image.shape[:2]
                    long_side = max(h, w)
                    if long_side > sam_long_side:
                        scale = sam_long_side / long_side
                        new_w, new_h = int(w * scale), int(h * scale)
                        render_image = cv2.resize(render_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    try:
                        # 获取FastSAM参数
                        fastsam_conf = self.fastsam_conf_slider.value() / 100.0
                        fastsam_iou = self.fastsam_iou_slider.value() / 100.0
                        
                        if is_point_prompt:
                            if sam_model_type.startswith('fastsam'):
                                prompts_2d = porject_to_2d(view, prompts_3d)
                                prompts_2d_tensor = torch.tensor(prompts_2d, dtype=torch.float32)
                                sam_mask_np = fastsam_point_prompt(self.predictor, render_image, prompts_2d_tensor, mask_id, conf=fastsam_conf, iou=fastsam_iou)
                            else:
                                prompts_2d = porject_to_2d(view, prompts_3d)
                                # 使用多个点提示进行SAM分割
                                self.predictor.set_image(render_image)
                                masks, scores, logits = self.predictor.predict(
                                    point_coords=input_points,
                                    point_labels=input_labels,
                                    multimask_output=True,
                                )
                                sam_mask_np = masks[mask_id if mask_id < len(masks) else np.argmax(scores)].astype(np.uint8)
                        else:
                            if sam_model_type.startswith('fastsam'):
                                sam_mask_np = fastsam_text_prompt(self.predictor, render_image, text_prompt, mask_id, conf=fastsam_conf, iou=fastsam_iou)
                            else:
                                sam_mask_np = text_prompting(self.predictor, render_image, text_prompt, mask_id)
                    except Exception as e:
                        self.log(f"视图 {i+1} 分割失败: {str(e)}", "error")
                        continue
                    
                    # Mask后处理（如果启用）
                    if self.enable_postprocess_check.isChecked():
                        sam_mask_np = post_process_mask(sam_mask_np, morph_kernel_size=3, smooth=True)
                        # 归一化回0-1范围
                        if sam_mask_np.max() > 1.0:
                            sam_mask_np = sam_mask_np / 255.0
                    
                    # 转换为tensor
                    sam_mask = torch.from_numpy(sam_mask_np).to('cuda' if torch.cuda.is_available() else 'cpu')
                    if len(sam_mask.shape) != 2:
                        sam_mask = sam_mask.squeeze(-1)
                    sam_mask = sam_mask.long()
                    
                    # 保存SAM mask
                    sam_masks.append(sam_mask)
                    
                    # 生成点掩码
                    point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)
                    multiview_masks.append(point_mask.unsqueeze(-1))
                    
                    # 调试模式输出
                    if self.debug_mode_check.isChecked():
                        valid_points = torch.sum(point_mask >= 0)
                        selected_points = torch.sum(point_mask == 1)
                        self.log(f"  视图{i+1}: 有效点{valid_points}, 选中点{selected_points}")
                
                # 根据模型类型调整阈值
                if sam_model_type == 'vit_b':
                    # ViT-B模型精度较低，使用更低的阈值
                    adjusted_threshold = max(0.3, threshold - 0.2)
                    self.log(f"ViT-B模型检测，调整投票阈值: {threshold} -> {adjusted_threshold}")
                elif sam_model_type == 'vit_l':
                    # ViT-L模型中等精度，稍微降低阈值
                    adjusted_threshold = max(0.4, threshold - 0.1)
                    self.log(f"ViT-L模型检测，调整投票阈值: {threshold} -> {adjusted_threshold}")
                elif sam_model_type.startswith('fastsam'):
                    # FastSAM模型，使用中等阈值
                    adjusted_threshold = max(0.4, threshold - 0.1)
                    self.log(f"FastSAM模型检测，调整投票阈值: {threshold} -> {adjusted_threshold}")
                else:
                    # ViT-H模型高精度，使用原始阈值
                    adjusted_threshold = threshold
                
                # 多视角投票
                self.update_status("正在整合分割结果...", 85)
                _, final_mask = ensemble(multiview_masks, threshold=adjusted_threshold)
                
                # 保存结果
                self.final_mask = final_mask
                self.sam_masks = sam_masks
                
                # 保存当前迭代次数，用于后续查看时使用
                current_iteration = int(self.iteration_edit.text())
                self.segmentation_iteration = current_iteration
                
                self.update_status("分割完成", 100)
                self.log(f"分割完成，选中 {len(final_mask)} 个高斯点")
                self.log(f"最终内存状态: {MemoryManager.get_gpu_memory_info()}")
                
                # 调试信息
                if len(final_mask) == 0:
                    self.log("⚠️ 警告: 没有选中任何高斯点", "warning")
                    self.log("可能原因:", "warning")
                    self.log("1. 点击位置不在有效区域内", "warning")
                    self.log("2. 投票阈值过高", "warning")
                    self.log("3. SAM模型分割质量不佳", "warning")
                    self.log("建议:", "warning")
                    self.log("- 尝试降低投票阈值", "warning")
                    self.log("- 尝试不同的点击位置", "warning")
                    self.log("- 检查SAM模型选择", "warning")
                else:
                    self.log(f"✓ 成功分割，选中 {len(final_mask)} 个高斯点", "info")
                
                # 显示分割结果
                self.display_segmentation_result()
                
                # 更新3D渲染状态
                self.set_text_signal.emit(self.render_status_label, "分割完成，可以显示3D模型")
                
                # 更新3D渲染器显示分割结果
                self.update_3d_viewer_with_segmentation()
                
                # 更新系统信息
                self.update_system_info()
                
            except Exception as e:
                self.update_status("分割失败", 0)
                self.log(f"分割失败: {str(e)}", "error")
                traceback.print_exc()
                self.show_message("错误", f"分割失败: {str(e)}", "critical")
        
        threading.Thread(target=segmentation_thread, daemon=True).start()
    
    def display_segmentation_result(self, view_idx=None):
        """显示分割结果 - 显示ensemble后的最终结果，而不是单个视图的SAM mask"""
        if not hasattr(self, 'final_mask') or not self.model_loaded:
            return
            
        # 如果没有指定视图索引，使用当前视图索引
        if view_idx is None:
            view_idx = self.current_view_idx
            
        try:
            # 显示指定视图的分割结果
            view = self.cameras[view_idx]
            
            with torch.no_grad():
                render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
            
            # 使用final_mask生成当前视图的2D投影掩码
            h, w = render_image.shape[:2]
            
            # 将final_mask转换为当前视图的2D投影掩码
            xyz = self.gaussians.get_xyz
            final_mask_device = self.final_mask.to(xyz.device) if isinstance(self.final_mask, torch.Tensor) else torch.tensor(self.final_mask, device=xyz.device)
            
            # 投影所有高斯点到当前视图
            xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=1)
            point_image = porject_to_2d(view, xyz_h).long()
            
            # 创建视图掩码：只显示final_mask中的点
            view_mask = np.zeros((h, w), dtype=np.uint8)
            valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
            valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
            valid_mask = valid_x & valid_y
            
            # 将valid_mask转换为numpy数组，以便与numpy数组进行操作
            valid_mask_np = valid_mask.cpu().numpy() if isinstance(valid_mask, torch.Tensor) else np.array(valid_mask)
            
            # final_mask是索引数组（ensemble函数返回的indices_mask）
            if isinstance(final_mask_device, torch.Tensor):
                final_mask_np = final_mask_device.cpu().numpy()
            else:
                final_mask_np = np.array(final_mask_device)
            
            # 判断是布尔掩码还是索引数组
            is_bool_mask = final_mask_np.dtype == bool
            
            if is_bool_mask:
                # 布尔掩码：直接使用或调整大小
                if len(final_mask_np) != len(xyz):
                    self.log(f"警告: 布尔掩码大小不匹配（掩码: {len(final_mask_np)}, 模型: {len(xyz)}），正在调整", "warning")
                    
                    # 创建正确大小的掩码
                    selected_bool = np.zeros(len(xyz), dtype=bool)
                    if len(final_mask_np) > len(xyz):
                        # 掩码太大，截断
                        selected_bool[:] = final_mask_np[:len(xyz)]
                    else:
                        # 掩码太小，复制可用部分
                        selected_bool[:len(final_mask_np)] = final_mask_np
                else:
                    selected_bool = final_mask_np
            else:
                # 索引数组：创建布尔数组
                selected_bool = np.zeros(len(xyz), dtype=bool)
                if len(final_mask_np) > 0:
                    # 过滤掉超出范围的索引，避免越界错误
                    valid_indices = final_mask_np[final_mask_np < len(xyz)]
                    invalid_count = len(final_mask_np) - len(valid_indices)
                    
                    if invalid_count > 0:
                        self.log(f"警告: {invalid_count} 个索引超出范围（最大索引: {final_mask_np.max()}, 模型大小: {len(xyz)}），已自动过滤", "warning")
                    
                    if len(valid_indices) > 0:
                        selected_bool[valid_indices] = True
                    else:
                        self.log("警告: 没有有效的分割索引", "warning")
            
            # 标记在final_mask中且在有效范围内的点
            valid_selected = valid_mask_np & selected_bool
            if valid_selected.any():
                point_image_np = point_image.cpu().numpy() if isinstance(point_image, torch.Tensor) else point_image
                selected_y = np.clip(point_image_np[valid_selected, 1], 0, h-1)
                selected_x = np.clip(point_image_np[valid_selected, 0], 0, w-1)
                view_mask[selected_y.astype(int), selected_x.astype(int)] = 255
            
            # 创建彩色叠加图像
            colored_mask = np.zeros_like(render_image)
            
            # 显示最终ensemble结果（红色）
            colored_mask[view_mask > 0] = [255, 0, 0]
            
            # 可选：同时显示单个视图的SAM mask作为对比（淡蓝色，半透明）
            if hasattr(self, 'sam_masks') and view_idx < len(self.sam_masks):
                sam_mask = self.sam_masks[view_idx]
                sam_mask_np = sam_mask.cpu().numpy() if isinstance(sam_mask, torch.Tensor) else sam_mask
                if sam_mask_np.shape != (h, w):
                    sam_mask_np = _resize_mask_torch(torch.from_numpy(sam_mask_np) if not isinstance(sam_mask_np, torch.Tensor) else sam_mask_np, h, w).numpy()
                
                # 将SAM mask叠加为淡蓝色（半透明）
                sam_mask_bool = sam_mask_np > 0.5
                colored_mask[sam_mask_bool] = np.maximum(colored_mask[sam_mask_bool], [100, 150, 255])  # 淡蓝色
            
            # 混合原始图像和掩码
            alpha = 0.4  # 掩码透明度
            blend_image = render_image.copy()
            mask_area = colored_mask.max(axis=2) > 0
            blend_image[mask_area] = (1 - alpha) * blend_image[mask_area] + alpha * colored_mask[mask_area]
            blend_image = np.clip(blend_image, 0, 255).astype(np.uint8)
            
            # 显示结果
            self.segmentation_canvas.set_image(blend_image)
            
            self.log(f"显示视图 {view_idx} 的分割结果")
            
        except Exception as e:
            self.log(f"显示分割结果失败: {str(e)}", "error")
    
    def save_results(self):
        """保存结果"""
        if not hasattr(self, 'final_mask') or self.final_mask is None:
            self.show_message("警告", "没有可保存的分割结果", "warning")
            return
        
        path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if path:
            try:
                # 保存分割结果
                save_gs(self.gaussians, self.final_mask, path)
                self.log(f"分割结果已保存到: {path}", "info")
                self.show_message("成功", "分割结果已保存", "info")
            except Exception as e:
                self.log(f"保存失败: {str(e)}", "error")
                self.show_message("错误", f"保存失败: {str(e)}", "critical")
    
    # 预处理相关方法
    def load_original_images(self, image_dir):
        """加载原始图像用于预处理"""
        try:
            self.log(f"正在加载原始图像: {image_dir}")
            
            # 支持的图像格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            
            # 获取所有图像文件
            image_files = []
            for file in os.listdir(image_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file)
            
            if not image_files:
                self.log("未找到图像文件", "warning")
                return
            
            # 按文件名排序
            image_files.sort()
            
            # 加载图像
            self.original_images = {}
            for i, filename in enumerate(image_files):
                image_path = os.path.join(image_dir, filename)
                try:
                    # 使用PIL加载图像
                    image = Image.open(image_path)
                    # 转换为RGB格式
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # 转换为numpy数组
                    image_array = np.array(image)
                    self.original_images[filename] = image_array
                    
                    if i == 0:
                        # 显示第一张图像
                        self.display_original_image(image_array, filename)
                        
                except Exception as e:
                    self.log(f"加载图像失败 {filename}: {str(e)}", "warning")
                    continue
            
            self.log(f"成功加载 {len(self.original_images)} 张原始图像")
            
            # 如果启用了预处理，初始化预处理相关功能
            if self.enable_preprocess_check.isChecked():
                self.init_preprocess_features()
                
        except Exception as e:
            self.log(f"加载原始图像失败: {str(e)}", "error")
    
    def display_original_image(self, image_array, filename):
        """显示原始图像"""
        try:
            # 直接显示图像（Qt的ImageCanvas会自动处理缩放）
            self.original_canvas.set_image(image_array)
            self.current_image = image_array
            self.log(f"显示原始图像: {filename}")
            
        except Exception as e:
            self.log(f"显示原始图像失败: {str(e)}", "error")
    
    def init_preprocess_features(self):
        """初始化预处理特征"""
        if not self.original_images:
            return
        
        def init_thread():
            try:
                self.log("正在初始化预处理SAM特征...")
                
                # 初始化SAM预测器
                sam_model = self.preprocess_sam_model_combo.currentText()
                sam_long_side = int(self.preprocess_sam_long_side_combo.currentText())
                
                # 获取SAM模型路径
                sam_model_paths = {
                    'vit_h': 'sam_vit_h_4b8939.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth', 
                    'vit_b': 'sam_vit_b_01ec64.pth',
                    'fastsam_s': 'FastSAM-s.pt',
                    'fastsam_x': 'FastSAM-x.pt'
                }
                
                sam_ckpt_filename = sam_model_paths.get(sam_model, 'sam_vit_b_01ec64.pth')
                sam_ckpt_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
                
                if not os.path.isfile(sam_ckpt_path):
                    self.log(f"SAM模型文件不存在: {sam_ckpt_path}", "error")
                    return
                
                # 初始化SAM预测器
                if sam_model.startswith('fastsam'):
                    self.preprocess_predictor = init_fastsam_predictor(sam_ckpt_path, model_type=sam_model)
                else:
                    self.preprocess_predictor = init_sam_predictor(sam_ckpt_path, model_type=sam_model)
                
                # 提取SAM特征
                self.preprocess_sam_features = {}
                for filename, image_array in self.original_images.items():
                    try:
                        # 调整图像大小
                        h, w = image_array.shape[:2]
                        long_side = max(h, w)
                        if long_side > sam_long_side:
                            scale = sam_long_side / long_side
                            new_w, new_h = int(w * scale), int(h * scale)
                            resized_image = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        else:
                            resized_image = image_array
                        
                        # 设置图像并提取特征
                        if sam_model.startswith('fastsam'):
                            # FastSAM doesn't need feature extraction - store image for later use
                            self.preprocess_sam_features[filename] = resized_image
                        else:
                            self.preprocess_predictor.set_image(resized_image)
                            feats = self.preprocess_predictor.features
                            if isinstance(feats, torch.Tensor):
                                feats = feats.cpu()
                            self.preprocess_sam_features[filename] = feats
                        
                    except Exception as e:
                        self.log(f"提取SAM特征失败 {filename}: {str(e)}", "warning")
                        continue
                
                self.log(f"预处理SAM特征初始化完成，处理了 {len(self.preprocess_sam_features)} 张图像")
                
            except Exception as e:
                self.log(f"初始化预处理特征失败: {str(e)}", "error")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def on_preprocess_canvas_click(self, x, y):
        """预处理画布点击事件"""
        if not self.original_images or self.current_image is None:
            return
        
        try:
            # 获取点击坐标（已经在图像坐标中）
            # Qt版本中，ImageCanvas已经处理了坐标转换
            canvas = self.original_canvas
            if canvas.image is None:
                return
            
            # 获取图像尺寸
            img_h, img_w = canvas.image.shape[:2]
            
            # 将窗口坐标转换为图像坐标
            img_x = int((x - canvas.offset_x) / canvas.zoom)
            img_y = int((y - canvas.offset_y) / canvas.zoom)
            
            # 确保在图像范围内
            img_x = max(0, min(img_w - 1, img_x))
            img_y = max(0, min(img_h - 1, img_y))
            
            # 添加点提示
            label = self.preprocess_current_point_type
            self.preprocess_points.append((img_x, img_y, label))
            canvas.add_point(x, y, label)
            self.update_preprocess_points_display()
            
            point_type_str = "正点" if label == 1 else "负点"
            self.log(f"添加预处理{point_type_str}: ({img_x}, {img_y})")
            
        except Exception as e:
            self.log(f"预处理点击处理失败: {str(e)}", "error")
    
    def run_preprocess_segmentation(self):
        """执行预处理分割"""
        if not self.original_images or not self.preprocess_predictor:
            self.log("请先加载原始图像并初始化预处理特征", "error")
            return
        
        def seg_thread():
            try:
                self.log("开始执行预处理分割...")
                
                # 获取预处理参数
                prompt_type = "point" if self.preprocess_point_prompt_radio.isChecked() else "text"
                mask_id = self.preprocess_mask_id
                
                if prompt_type == "point":
                    if not self.preprocess_points:
                        self.log("请先添加预处理点提示", "error")
                        return
                    
                    # 分离正点和负点
                    positive_points = [(x, y) for x, y, label in self.preprocess_points if label == 1]
                    negative_points = [(x, y) for x, y, label in self.preprocess_points if label == 0]
                    
                    if not positive_points:
                        self.log("请至少添加一个正点", "error")
                        return
                    
                    input_points = np.array(positive_points)
                    input_labels = np.ones(len(positive_points))
                    
                    if negative_points:
                        neg_points = np.array(negative_points)
                        neg_labels = np.zeros(len(negative_points))
                        input_points = np.vstack([input_points, neg_points])
                        input_labels = np.concatenate([input_labels, neg_labels])
                    
                    self.log(f"使用 {len(positive_points)} 个正点和 {len(negative_points)} 个负点进行预处理分割")
                    
                else:  # text prompt
                    text_prompt = self.preprocess_text_edit.text()
                    if not text_prompt:
                        self.log("请输入文本提示", "error")
                        return
                    
                    input_points = None
                    input_labels = None
                    self.log(f"使用文本提示进行预处理分割: {text_prompt}")
                
                # 对每张图像执行分割
                self.preprocess_masks = {}
                sam_model = self.preprocess_sam_model_combo.currentText()
                sam_long_side = int(self.preprocess_sam_long_side_combo.currentText())
                
                for filename, image_array in self.original_images.items():
                    try:
                        # 调整图像大小
                        h, w = image_array.shape[:2]
                        long_side = max(h, w)
                        if long_side > sam_long_side:
                            scale = sam_long_side / long_side
                            new_w, new_h = int(w * scale), int(h * scale)
                            resized_image = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        else:
                            resized_image = image_array
                        
                        # 设置图像
                        if sam_model.startswith('fastsam'):
                            # FastSAM doesn't need set_image - we'll run inference directly
                            pass
                        else:
                            self.preprocess_predictor.set_image(resized_image)
                        
                        if prompt_type == "point":
                            # 调整点坐标到调整后的图像尺寸
                            scaled_points = input_points * (new_w / w) if new_w != w else input_points
                            scaled_points[:, 1] = scaled_points[:, 1] * (new_h / h) if new_h != h else scaled_points[:, 1]
                            
                            # 使用多个点提示进行SAM分割
                            if sam_model.startswith('fastsam'):
                                scaled_points_tensor = torch.tensor(scaled_points, dtype=torch.float32)
                                sam_mask_np = fastsam_point_prompt(self.preprocess_predictor, resized_image, scaled_points_tensor, mask_id)
                            else:
                                sam_mask_np = self.multi_point_sam_prompt(
                                    self.preprocess_predictor, resized_image, scaled_points, input_labels, mask_id
                                )
                        else:
                            # 文本提示分割
                            if sam_model.startswith('fastsam'):
                                sam_mask_np = fastsam_text_prompt(self.preprocess_predictor, resized_image, text_prompt, mask_id)
                            else:
                                sam_mask_np = text_prompting(self.preprocess_predictor, resized_image, text_prompt, mask_id)
                        
                        # 将mask调整回原始尺寸
                        if resized_image.shape[:2] != (h, w):
                            sam_mask_np = cv2.resize(sam_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        self.preprocess_masks[filename] = sam_mask_np
                        
                    except Exception as e:
                        self.log(f"预处理分割失败 {filename}: {str(e)}", "warning")
                        continue
                
                self.log(f"预处理分割完成，处理了 {len(self.preprocess_masks)} 张图像")
                
                # 显示第一张图像的分割结果
                if self.preprocess_masks:
                    first_filename = list(self.preprocess_masks.keys())[0]
                    mask = self.preprocess_masks[first_filename]
                    self.display_preprocess_result(mask, first_filename)
                
            except Exception as e:
                self.log(f"预处理分割失败: {str(e)}", "error")
        
        threading.Thread(target=seg_thread, daemon=True).start()
    
    def display_preprocess_result(self, mask, filename):
        """显示预处理分割结果"""
        try:
            # 获取原始图像
            if filename not in self.original_images:
                return
            
            original_image = self.original_images[filename]
            
            # 创建可视化图像
            vis_image = original_image.copy()
            
            # 将mask转换为彩色叠加
            mask_colored = np.zeros_like(original_image)
            mask_colored[mask > 0] = [0, 255, 0]  # 绿色表示分割区域
            
            # 混合原始图像和mask
            alpha = 0.3
            vis_image = cv2.addWeighted(vis_image, 1-alpha, mask_colored, alpha, 0)
            
            # 显示结果
            self.display_original_image(vis_image, f"{filename}_preprocess")
            
        except Exception as e:
            self.log(f"显示预处理结果失败: {str(e)}", "error")
    
    def on_preprocess_toggle(self):
        """预处理开关切换"""
        enabled = self.enable_preprocess_check.isChecked()
        self.preprocess_prompt_group.setVisible(enabled)
    
    def on_preprocess_prompt_type_change(self):
        """预处理提示方式改变"""
        is_point = self.preprocess_point_prompt_radio.isChecked()
        # 在Qt中可以通过布局控制显示/隐藏
        # 这里简化处理，只更新状态
        self.preprocess_text_edit.setEnabled(not is_point)
    
    def clear_all_preprocess_points(self):
        """清除所有预处理点"""
        self.preprocess_points = []
        self.preprocess_points_listbox.clear()
        self.log("已清除所有预处理点提示", "info")
    
    def remove_last_preprocess_point(self):
        """移除最后一个预处理点"""
        if self.preprocess_points:
            self.preprocess_points.pop()
            self.update_preprocess_points_display()
            self.log("已移除最后一个预处理点", "info")
    
    def update_preprocess_points_display(self):
        """更新预处理点列表显示"""
        self.preprocess_points_listbox.clear()
        for i, (x, y, point_type) in enumerate(self.preprocess_points):
            point_type_str = "正点" if point_type == 1 else "负点"
            self.preprocess_points_listbox.addItem(f"{i+1}. {point_type_str} ({x:.0f}, {y:.0f})")
    
    def validate_preprocess_params(self):
        """验证预处理参数"""
        try:
            if not self.enable_preprocess_check.isChecked():
                return True
            
            prompt_type = "point" if self.preprocess_point_prompt_radio.isChecked() else "text"
            
            if prompt_type == "point":
                if not self.preprocess_points:
                    raise ValueError("请先添加预处理点提示")
            else:  # text
                text_prompt = self.preprocess_text_edit.text()
                if not text_prompt:
                    raise ValueError("请输入预处理文本提示")
            
            return True
        except ValueError as e:
            self.show_message("参数错误", str(e), "critical")
            return False
    
    # 视图控制方法
    def change_view(self):
        """改变视图"""
        try:
            idx = self.view_spinbox.value()
            if 0 <= idx < len(self.cameras):
                self.current_view_idx = idx
                self.display_current_view()
        except Exception as e:
            self.log(f"切换视图失败: {str(e)}", "error")
    
    def prev_view(self):
        """上一视图"""
        if self.current_view_idx > 0:
            self.current_view_idx -= 1
            self.view_spinbox.setValue(self.current_view_idx)
    
    def next_view(self):
        """下一视图"""
        if self.current_view_idx < len(self.cameras) - 1:
            self.current_view_idx += 1
            self.view_spinbox.setValue(self.current_view_idx)
    
    # 3D可视化方法
    def show_3d_model(self):
        """显示3D模型（提示使用SIBR查看器）"""
        # 检查是否已加载模型
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.show_message("错误", "请先加载3DGS模型", "critical")
            return
        
        if not hasattr(self, 'gaussians') or self.gaussians is None:
            self.show_message("错误", "模型数据未加载", "critical")
            return
        
        try:
            # 检查是否有分割结果
            has_segmentation = hasattr(self, 'final_mask') and self.final_mask is not None
            
            # 获取所有高斯点
            xyz = self.gaussians.get_xyz.detach().cpu().numpy()
            
            # 根据是否有分割结果决定显示哪些点
            if has_segmentation:
                # 只显示分割的点
                display_xyz = xyz[self.final_mask]
                is_segmented = True
            else:
                # 显示所有原始点
                display_xyz = xyz
                is_segmented = False
            
            # 更新状态和统计信息
            point_count = len(display_xyz)
            if is_segmented:
                status_text = f"3D模型已准备（分割后，{point_count:,}个点）"
            else:
                status_text = f"3D模型已准备（原始模型，{point_count:,}个点）"
            self.set_text_signal.emit(self.render_status_label, status_text)
            if hasattr(self, 'point_count_label'):
                self.set_text_signal.emit(self.point_count_label, f"{point_count:,} 个点")
            
            self.log(f"3D模型已准备，共{point_count:,}个高斯点。请使用SIBR查看器查看3D模型。")
            self.show_message("提示", "请使用上方的'启动SIBR查看器'按钮来查看3D模型", "information")
            
        except Exception as e:
            self.log(f"处理3D模型失败: {str(e)}", "error")
            self.show_message("错误", f"处理3D模型失败: {str(e)}", "critical")
            # 更新状态为错误
            self.set_text_signal.emit(self.render_status_label, "3D模型处理失败")
            if hasattr(self, 'point_count_label'):
                self.set_text_signal.emit(self.point_count_label, "N/A")
    
    def sh_to_rgb(self, sh_coeffs):
        """将球谐系数转换为RGB颜色"""
        rgb = 0.28209479177387814 * sh_coeffs[:, 0]
        rgb = np.clip(rgb, 0, 1)
        return rgb
    
    
    def launch_3d_viewer(self):
        """启动3D查看器"""
        self.log("开始启动外部3D查看器...")
        
        # 检查是否已加载模型
        if not self.model_loaded:
            self.log("错误: 请先加载3DGS模型", "error")
            self.show_message("错误", "请先加载3DGS模型", "critical")
            return
        
        if not os.path.exists(self.sibr_viewer_path):
            self.log(f"错误: SIBR查看器未找到: {self.sibr_viewer_path}", "error")
            self.show_message("错误", f"SIBR查看器未找到: {self.sibr_viewer_path}", "critical")
            return
        
        try:
            # 使用当前加载的模型目录
            model_path = self.model_path_edit.text()
            
            if not model_path or not os.path.exists(model_path):
                self.log("错误: 模型路径无效或不存在", "error")
                self.show_message("错误", "请先加载3DGS模型", "critical")
                return
            
            self.log(f"初始模型路径: {model_path}")
            
            # 智能查找包含必要文件的正确目录
            required_files = ['cfg_args', 'cameras.json']
            
            def find_model_directory(base_path):
                """递归查找包含必要文件的模型目录"""
                # 首先检查当前目录
                if all(os.path.exists(os.path.join(base_path, f)) for f in required_files):
                    return base_path
                
                # 检查是否是output目录，如果是，查找最新的子目录
                if os.path.basename(base_path) == 'output' or 'output' in base_path:
                    subdirs = []
                    try:
                        for item in os.listdir(base_path):
                            item_path = os.path.join(base_path, item)
                            if os.path.isdir(item_path):
                                # 检查这个子目录是否包含必要文件
                                if all(os.path.exists(os.path.join(item_path, f)) for f in required_files):
                                    # 获取修改时间，找最新的
                                    mtime = os.path.getmtime(item_path)
                                    subdirs.append((mtime, item_path))
                        
                        if subdirs:
                            # 返回最新的目录
                            subdirs.sort(reverse=True)
                            return subdirs[0][1]
                    except:
                        pass
                
                # 检查父目录
                parent_path = os.path.dirname(base_path)
                if parent_path and parent_path != base_path:
                    if all(os.path.exists(os.path.join(parent_path, f)) for f in required_files):
                        return parent_path
                
                # 检查常见的子目录模式
                common_subdirs = ['', 'point_cloud', os.path.basename(base_path)]
                for subdir in common_subdirs:
                    if subdir:
                        test_path = os.path.join(base_path, subdir)
                    else:
                        test_path = base_path
                    
                    if os.path.exists(test_path):
                        if all(os.path.exists(os.path.join(test_path, f)) for f in required_files):
                            return test_path
                
                return None
            
            # 查找正确的模型目录
            correct_model_path = find_model_directory(model_path)
            
            if correct_model_path is None:
                # 未找到，给出详细错误信息
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
                
                error_msg = f"无法找到包含必要文件的模型目录\n\n"
                error_msg += f"当前路径: {model_path}\n"
                error_msg += f"缺少文件: {missing_files}\n\n"
                error_msg += f"提示:\n"
                error_msg += f"1. 确保路径指向3DGS训练输出的根目录\n"
                error_msg += f"2. 目录下应包含 cfg_args 和 cameras.json 文件\n"
                error_msg += f"3. 通常路径格式为: .../output/<scene_name>\n"
                
                # 尝试列出目录内容帮助调试
                try:
                    if os.path.exists(model_path):
                        items = os.listdir(model_path)
                        error_msg += f"\n当前目录内容: {', '.join(items[:10])}"
                        if len(items) > 10:
                            error_msg += f"... (共{len(items)}项)"
                except:
                    pass
                
                self.log(error_msg, "error")
                self.show_message("错误", error_msg, "critical")
                return
            
            # 如果找到了不同的路径，更新并通知用户
            if correct_model_path != model_path:
                self.log(f"✓ 自动找到正确的模型目录: {correct_model_path}", "info")
                model_path = correct_model_path
                # 可选：更新GUI中的路径显示
                # self.model_path_edit.setText(model_path)
            else:
                self.log(f"✓ 模型目录验证通过: {model_path}")
            
            # 获取可用的迭代次数
            available_iterations = self.get_available_iterations(model_path)
            if not available_iterations:
                self.show_message("错误", "未找到任何可用的迭代结果", "critical")
                return
            
            # 让用户选择迭代次数（总是显示选择对话框）
            selected_iteration = self.select_iteration_dialog(model_path)
            if selected_iteration is None:
                return
            self.log(f"用户选择的迭代次数: {selected_iteration}")
            
            # 创建point_cloud目录
            point_cloud_dir = os.path.join(model_path, "point_cloud")
            iteration_dir = os.path.join(point_cloud_dir, f"iteration_{selected_iteration}")
            
            # 检查迭代目录是否存在
            if not os.path.exists(iteration_dir):
                self.show_message("错误", f"迭代目录不存在: {iteration_dir}", "critical")
                return
            
            # 检查是否有分割结果（包括SAGA分割）
            has_segmentation = False
            segmentation_source = None
            
            if hasattr(self, 'final_mask') and self.final_mask is not None:
                # 验证分割结果是否有效
                if isinstance(self.final_mask, torch.Tensor):
                    mask_size = self.final_mask.sum().item() if self.final_mask.dtype == torch.bool else self.final_mask.numel()
                else:
                    mask_size = len(self.final_mask) if hasattr(self.final_mask, '__len__') else 0
                
                if mask_size > 0:
                    has_segmentation = True
                    
                    # 判断分割来源：根据mask类型和大小
                    mask_length = self.final_mask.numel() if isinstance(self.final_mask, torch.Tensor) else len(self.final_mask)
                    is_bool_mask = self.final_mask.dtype == torch.bool if isinstance(self.final_mask, torch.Tensor) else False
                    
                    # 获取不同gaussians的点数
                    saga_points = self.saga_module.gaussians._xyz.shape[0] if (SAGA_AVAILABLE and self.saga_module and self.saga_module.gaussians is not None) else -1
                    sags_points = self.gaussians._xyz.shape[0] if (self.gaussians is not None) else -1
                    
                    # 判断逻辑：
                    # 1. SAGA分割产生布尔掩码，长度=总点数
                    # 2. SAGS分割产生索引数组，长度=选中的点数（远小于总点数）
                    if is_bool_mask and saga_points > 0 and mask_length == saga_points:
                        segmentation_source = "SAGA"
                        self.log(f"检测到SAGA分割 (布尔mask，大小: {mask_length}, SAGA点数: {saga_points})")
                    elif not is_bool_mask and sags_points > 0 and mask_length < sags_points:
                        # 索引数组，且长度远小于总点数
                        segmentation_source = "SAGS"
                        self.log(f"检测到SAGS分割 (索引数组，大小: {mask_length}, SAGS点数: {sags_points})")
                    elif is_bool_mask and sags_points > 0 and mask_length == sags_points:
                        # 布尔掩码匹配SAGS点数（可能是SAGS的布尔掩码版本）
                        segmentation_source = "SAGS"
                        self.log(f"检测到SAGS分割 (布尔mask，大小: {mask_length}, SAGS点数: {sags_points})")
                    else:
                        # 无法确定来源，根据是否为布尔类型猜测
                        if is_bool_mask:
                            segmentation_source = "SAGA"
                            self.log(f"假设为SAGA分割 (布尔mask，大小: {mask_length})", "warning")
                        else:
                            segmentation_source = "SAGS"
                            self.log(f"假设为SAGS分割 (索引数组，大小: {mask_length})", "warning")
            
            if has_segmentation:
                # 检查迭代次数是否匹配（如果有设置）
                iteration_mismatch = False
                if hasattr(self, 'segmentation_iteration') and self.segmentation_iteration is not None:
                    if selected_iteration != self.segmentation_iteration:
                        iteration_mismatch = True
                        self.log(f"用户选择了不同的迭代次数 ({selected_iteration})，分割结果基于迭代 {self.segmentation_iteration}")
                        self.log("将显示原始模型，因为分割结果与选择的迭代次数不匹配")
                
                if not iteration_mismatch:
                    # 显示分割后的模型
                    self.log(f"检测到{segmentation_source}分割结果，显示分割后的模型...")
                    
                    if mask_size == 0:
                        self.log("⚠️ 警告: 分割结果为空，无法显示分割后的模型", "warning")
                        self.show_message("警告", "分割结果为空，无法显示分割后的模型。\n将显示原始模型。", "warning")
                        self.temp_ply_path = None
                    else:
                        # 备份原始point_cloud.ply文件
                        original_ply = os.path.join(iteration_dir, "point_cloud.ply")
                        backup_ply = os.path.join(iteration_dir, "point_cloud_original.ply")
                        if os.path.exists(original_ply) and not os.path.exists(backup_ply):
                            shutil.copy2(original_ply, backup_ply)
                            self.log("已备份原始point_cloud.ply文件")
                        
                        # 保存分割结果到point_cloud.ply（替换原始文件）
                        seg_ply_path = original_ply
                        try:
                            # 使用正确的gaussians对象（根据分割来源）
                            if segmentation_source == "SAGA":
                                gaussians_to_save = self.saga_module.gaussians
                            elif segmentation_source == "SAGS" or segmentation_source == "未知":
                                gaussians_to_save = self.gaussians
                            else:
                                # 默认使用主gaussians
                                gaussians_to_save = self.gaussians
                            
                            # 确保mask在CPU上
                            mask_to_save = self.final_mask
                            if isinstance(mask_to_save, torch.Tensor):
                                if mask_to_save.device.type == 'cuda':
                                    mask_to_save = mask_to_save.cpu()
                                # 注意：不强制转换为布尔类型，因为SAGS使用索引数组
                            
                            # 验证mask和gaussians大小是否匹配
                            gaussians_point_count = gaussians_to_save._xyz.shape[0]
                            mask_point_count = mask_to_save.numel() if isinstance(mask_to_save, torch.Tensor) else len(mask_to_save)
                            is_bool_mask = mask_to_save.dtype == torch.bool if isinstance(mask_to_save, torch.Tensor) else False
                            
                            # 只对布尔掩码验证大小匹配（索引数组的大小会小于总点数）
                            if is_bool_mask and gaussians_point_count != mask_point_count:
                                raise ValueError(
                                    f"布尔Mask和Gaussians大小不匹配：\n"
                                    f"  Gaussians点数: {gaussians_point_count}\n"
                                    f"  Mask大小: {mask_point_count}\n"
                                    f"  分割来源: {segmentation_source}\n"
                                    f"提示: 这通常是因为在不同的分割模式（SAGA/SAGS）之间切换导致的。\n"
                                    f"请清空之前的分割结果后再试。"
                                )
                            elif not is_bool_mask:
                                # 索引数组：验证索引范围
                                if isinstance(mask_to_save, torch.Tensor):
                                    max_idx = mask_to_save.max().item() if mask_to_save.numel() > 0 else -1
                                    min_idx = mask_to_save.min().item() if mask_to_save.numel() > 0 else 0
                                else:
                                    max_idx = max(mask_to_save) if len(mask_to_save) > 0 else -1
                                    min_idx = min(mask_to_save) if len(mask_to_save) > 0 else 0
                                
                                if max_idx >= gaussians_point_count:
                                    raise ValueError(
                                        f"索引数组超出范围：\n"
                                        f"  最大索引: {max_idx}\n"
                                        f"  Gaussians点数: {gaussians_point_count}\n"
                                        f"  分割来源: {segmentation_source}"
                                    )
                                if min_idx < 0:
                                    raise ValueError(f"索引数组包含负数索引: {min_idx}")
                            
                            save_gs(gaussians_to_save, mask_to_save, seg_ply_path)
                            self.log(f"{segmentation_source}分割结果已保存到: {seg_ply_path}，包含 {mask_size} 个点")
                            
                            # 验证保存的文件是否有效
                            if not os.path.exists(seg_ply_path) or os.path.getsize(seg_ply_path) == 0:
                                raise ValueError("保存的PLY文件为空或不存在")
                            
                            self.temp_ply_path = seg_ply_path  # 保存分割结果文件路径
                        except Exception as e:
                            error_msg = f"保存{segmentation_source}分割结果失败: {str(e)}"
                            self.log(error_msg, "error")
                            traceback.print_exc()
                            self.show_message("错误", f"{error_msg}\n将显示原始模型。", "critical")
                            self.temp_ply_path = None
                else:
                    self.temp_ply_path = None
            else:
                # 没有分割结果，显示原始模型
                self.log("未检测到分割结果，显示原始模型...")
                self.temp_ply_path = None
            
            # 在启动SIBR查看器前，验证PLY文件
            if self.temp_ply_path and os.path.exists(self.temp_ply_path):
                ply_size = os.path.getsize(self.temp_ply_path)
                self.log(f"验证分割后的PLY文件: {self.temp_ply_path}, 大小: {ply_size} 字节")
                
                if ply_size == 0:
                    error_msg = "分割后的PLY文件为空，SIBR查看器可能无法加载"
                    self.log(f"错误: {error_msg}", "error")
                    self.show_message("错误", f"{error_msg}\n请检查分割结果。", "critical")
                    return
                
                # 尝试读取PLY文件头，验证文件格式
                try:
                    from plyfile import PlyData
                    plydata = PlyData.read(self.temp_ply_path)
                    num_vertices = len(plydata.elements[0])
                    self.log(f"PLY文件验证成功: 包含 {num_vertices} 个顶点")
                    
                    if num_vertices == 0:
                        error_msg = "PLY文件包含0个顶点，SIBR查看器无法显示"
                        self.log(f"错误: {error_msg}", "error")
                        self.show_message("错误", f"{error_msg}\n请检查分割结果。", "critical")
                        return
                except Exception as e:
                    error_msg = f"PLY文件格式错误: {str(e)}"
                    self.log(f"错误: {error_msg}", "error")
                    self.show_message("错误", f"{error_msg}\n请检查分割结果。", "critical")
                    return
            
            # 在启动SIBR查看器前，清理并释放GPU资源
            # 这是关键步骤，因为SIBR查看器需要独占访问CUDA设备
            self.log("正在清理GPU资源，为SIBR查看器释放CUDA设备...")
            try:
                # 强制清理所有GPU内存
                MemoryManager.clear_gpu_memory()
                
                # 如果有gaussians模型在GPU上，需要将其移到CPU或删除引用
                if self.gaussians is not None and torch.cuda.is_available():
                    try:
                        # 同步所有CUDA操作
                        torch.cuda.synchronize()
                        
                        # 清理所有CUDA缓存
                        torch.cuda.empty_cache()
                        
                        # 重置内存统计
                        torch.cuda.reset_peak_memory_stats()
                    except Exception as e:
                        self.log(f"清理模型GPU内存时出错: {str(e)}")
                
                # 执行Python垃圾回收，确保所有未使用的对象被释放
                gc.collect()
                
                # 再次清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 等待更长时间，让GPU完全释放
                # CUDA上下文释放需要一些时间
                wait_time = 1.0 if self.temp_ply_path else 0.5  # 分割后需要更长时间
                self.log(f"等待 {wait_time} 秒，让CUDA设备完全释放...")
                time.sleep(wait_time)
                
                # 再次检查CUDA设备是否可用
                if torch.cuda.is_available():
                    # 尝试创建一个小的测试张量来验证CUDA是否可用
                    try:
                        test_tensor = torch.zeros(1, device='cuda')
                        del test_tensor
                        torch.cuda.empty_cache()
                        self.log("CUDA设备验证成功，已准备就绪")
                    except RuntimeError as e:
                        self.log(f"警告: CUDA设备可能仍被占用: {str(e)}", "warning")
                        # 即使有警告，也尝试继续启动SIBR查看器
                
                self.log("GPU资源清理完成")
            except Exception as e:
                self.log(f"警告: GPU资源清理过程中出现错误: {str(e)}", "warning")
                # 继续尝试启动SIBR查看器
            
            # 启动SIBR查看器，使用-m参数指定模型目录，--iteration参数指定迭代次数
            cmd = [self.sibr_viewer_path, "-m", model_path, "--iteration", str(selected_iteration)]
            self.log(f"启动命令: {' '.join(cmd)}")
            self.log(f"模型路径: {model_path}")
            self.log(f"迭代次数: {selected_iteration}")
            if self.temp_ply_path:
                self.log(f"使用分割后的PLY文件: {self.temp_ply_path}")
            else:
                self.log("使用原始PLY文件")
            
            # 启动进程
            try:
                self.sibr_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,  # 改为PIPE以便捕获输出
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # 创建新的进程组
                )
                self.log("SIBR查看器进程已启动")
            except Exception as e:
                error_msg = f"启动SIBR查看器失败: {str(e)}"
                self.log(error_msg, "error")
                self.show_message("错误", error_msg, "critical")
                return
            
            # 等待一小段时间，检查进程是否立即退出
            time.sleep(1.0)  # 增加等待时间，给SIBR查看器更多时间初始化
            if self.sibr_process.poll() is not None:
                # 进程已退出，读取stderr和stdout查看错误信息
                stdout_output = ""
                stderr_output = ""
                try:
                    if self.sibr_process.stdout:
                        stdout_output = self.sibr_process.stdout.read().decode('utf-8', errors='ignore')
                    if self.sibr_process.stderr:
                        stderr_output = self.sibr_process.stderr.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    self.log(f"读取进程输出失败: {str(e)}")
                
                error_msg = f"SIBR查看器启动后立即退出 (退出码: {self.sibr_process.returncode})"
                
                # 检查是否是CUDA设备被占用的问题
                cuda_busy_error = False
                if stderr_output and ("CUDA" in stderr_output or "all CUDA-capable devices are busy" in stderr_output.lower()):
                    cuda_busy_error = True
                    error_msg += "\n\n⚠️ 检测到CUDA设备被占用错误！"
                    error_msg += "\n可能的原因:"
                    error_msg += "\n1. GPU仍被Python进程占用（分割操作后未完全释放）"
                    error_msg += "\n2. 其他程序正在使用GPU"
                    error_msg += "\n3. CUDA上下文未完全清理"
                    error_msg += "\n\n建议解决方案:"
                    error_msg += "\n1. 等待几秒钟后重试"
                    error_msg += "\n2. 关闭其他使用GPU的程序"
                    error_msg += "\n3. 重启应用程序"
                    error_msg += "\n4. 检查nvidia-smi查看GPU使用情况"
                
                if stderr_output:
                    error_msg += f"\n\n标准错误输出:\n{stderr_output}"
                if stdout_output:
                    error_msg += f"\n\n标准输出:\n{stdout_output}"
                
                # 添加诊断信息
                if self.temp_ply_path:
                    error_msg += f"\n\n诊断信息:"
                    error_msg += f"\n- 分割后的PLY文件: {self.temp_ply_path}"
                    if os.path.exists(self.temp_ply_path):
                        error_msg += f"\n- 文件大小: {os.path.getsize(self.temp_ply_path)} 字节"
                    else:
                        error_msg += "\n- 文件不存在！"
                
                # 如果是CUDA问题，添加额外的GPU状态信息
                if cuda_busy_error and torch.cuda.is_available():
                    try:
                        gpu_memory_info = MemoryManager.get_gpu_memory_info()
                        error_msg += f"\n\n当前GPU状态:\n{gpu_memory_info}"
                    except:
                        pass
                
                self.log(error_msg, "error")
                self.show_message("错误", error_msg, "critical")
                self.sibr_process = None
                return
            
            # 更新状态显示
            if self.temp_ply_path:
                self.set_text_signal.emit(self.render_status_label, "外部3D查看器已启动 (分割后模型)")
                self.log(f"SIBR查看器已启动，显示分割后的模型，模型目录: {model_path}")
            else:
                self.set_text_signal.emit(self.render_status_label, "外部3D查看器已启动 (原始模型)")
                self.log(f"SIBR查看器已启动，显示原始模型，模型目录: {model_path}")
            
        except Exception as e:
            self.log(f"启动3D查看器失败: {str(e)}", "error")
            self.log(traceback.format_exc(), "error")
            self.show_message("错误", f"启动3D查看器失败: {str(e)}", "critical")
    
    def stop_3d_viewer(self):
        """停止3D查看器并清理临时文件"""
        # 先停止进程
        if self.sibr_process:
            try:
                self.log("正在停止SIBR查看器进程...")
                self.sibr_process.terminate()
                self.sibr_process.wait(timeout=5)
                self.sibr_process = None
                self.log("SIBR查看器进程已停止")
            except Exception as e:
                self.log(f"停止3D查看器失败: {str(e)}", "error")
        
        # 清理临时文件并恢复原始模型
        if hasattr(self, 'temp_ply_path') and self.temp_ply_path:
            try:
                self.log("正在清理临时文件并恢复原始模型...")
                
                # 获取备份文件路径
                temp_dir = os.path.dirname(self.temp_ply_path)
                backup_ply = os.path.join(temp_dir, "point_cloud_original.ply")
                
                # 如果存在备份文件，恢复它
                if os.path.exists(backup_ply):
                    # 删除临时的分割文件（如果还存在）
                    if os.path.exists(self.temp_ply_path):
                        os.remove(self.temp_ply_path)
                        self.log(f"已删除临时分割文件: {self.temp_ply_path}")
                    
                    # 恢复原始文件
                    shutil.copy2(backup_ply, self.temp_ply_path)
                    self.log(f"已恢复原始文件: {self.temp_ply_path}")
                    
                    # 删除备份文件
                    os.remove(backup_ply)
                    self.log(f"已删除备份文件: {backup_ply}")
                else:
                    self.log("未找到备份文件，跳过恢复", "warning")
                
                # 清空临时路径记录
                self.temp_ply_path = None
                
            except Exception as e:
                self.log(f"清理临时文件时出错: {str(e)}", "error")
                traceback.print_exc()
        
        # 更新状态
        self.set_text_signal.emit(self.render_status_label, "外部3D查看器已停止")
        self.log("3D查看器已停止，临时文件已清理")
    
    # 浏览方法
    def browse_model_path(self):
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(self, "选择3DGS模型输出目录", self.model_path_edit.text())
        if path:
            self.model_path_edit.setText(path)
    
    def browse_source_path(self):
        """浏览数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择原始数据集目录", self.source_path_edit.text())
        if path:
            self.source_path_edit.setText(path)
    
    def browse_training_data_path(self):
        """浏览训练数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择训练数据集目录", self.training_data_path_edit.text())
        if path:
            self.training_data_path_edit.setText(path)
    
    def browse_training_output_path(self):
        """浏览训练输出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择训练输出目录", self.training_output_path_edit.text())
        if path:
            self.training_output_path_edit.setText(path)
    
    def browse_sog_data_path(self):
        """浏览SOG训练数据路径"""
        path = QFileDialog.getExistingDirectory(self, "选择SOG训练数据目录", self.sog_data_path_edit.text())
        if path:
            self.sog_data_path_edit.setText(path)
    
    def browse_sog_output_path(self):
        """浏览SOG输出路径"""
        path = QFileDialog.getExistingDirectory(self, "选择SOG输出目录", self.sog_output_path_edit.text())
        if path:
            self.sog_output_path_edit.setText(path)
    
    def browse_vocab_tree(self):
        """浏览词汇树文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择词汇树文件", self.colmap_vocab_tree_edit.text(), "Binary Files (*.bin);;All Files (*.*)")
        if path:
            self.colmap_vocab_tree_edit.setText(path)
    
    def on_colmap_input_type_change(self):
        """COLMAP输入类型改变"""
        is_video = self.colmap_video_radio.isChecked()
        self.video_params_group.setVisible(is_video)
    
    def browse_colmap_input_path(self):
        """浏览COLMAP输入路径"""
        if self.colmap_video_radio.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择COLMAP输入图像目录", self.colmap_input_path_edit.text())
        
        if path:
            self.colmap_input_path_edit.setText(path)
    
    # 训练相关方法
    def validate_training_params(self):
        """验证训练参数"""
        try:
            iterations = int(self.training_iterations_edit.text())
            if iterations <= 0:
                raise ValueError("迭代次数必须大于0")
            
            resolution = int(self.training_resolution_combo.currentText())
            if resolution not in [1, 2, 4, 8, -1]:
                raise ValueError("分辨率必须是1, 2, 4, 8或-1")
            
            lr = float(self.training_lr_edit.text())
            if lr <= 0:
                raise ValueError("学习率必须大于0")
            
            densify_from = int(self.densify_from_edit.text())
            densify_until = int(self.densify_until_edit.text())
            if densify_from < 0 or densify_until <= densify_from:
                raise ValueError("密化参数设置错误")
            
            test_interval = int(self.test_interval_edit.text())
            if test_interval <= 0:
                raise ValueError("测试间隔必须大于0")
            
            return True
        except ValueError as e:
            self.show_message("参数错误", str(e), "critical")
            return False
    
    def start_training(self):
        """开始训练3DGS模型"""
        if self.is_training:
            self.show_message("警告", "训练正在进行中", "warning")
            return
        
        # 验证训练参数
        if not self.validate_training_params():
            return
        
        data_path = self.training_data_path_edit.text()
        output_path = self.training_output_path_edit.text()
        
        if not data_path or not os.path.exists(data_path):
            self.show_message("错误", "请选择有效的训练数据路径", "critical")
            return
        
        # 检查数据集格式
        has_colmap = os.path.exists(os.path.join(data_path, 'sparse'))
        has_blender = os.path.exists(os.path.join(data_path, 'transforms_train.json'))
        
        if not (has_colmap or has_blender):
            self.show_message("错误", "数据集格式不正确。需要包含COLMAP的'sparse'目录或Blender的'transforms_train.json'文件", "critical")
            return
        
        # 自动根据输入数据路径创建同名输出文件夹
        # 获取输入数据文件夹名
        folder_name = os.path.basename(data_path.rstrip('/'))
        # 构建输出路径
        sags_dir = os.path.dirname(__file__)
        output_path = os.path.join(sags_dir, "gaussiansplatting", "output", folder_name)
        # 更新输出路径显示
        self.training_output_path_edit.setText(output_path)
        self.log(f"自动设置输出路径: {output_path}")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        def try_switch_to_undistorted(base_path: str) -> bool:
            """当检测到相机模型不被支持时，将数据目录切换为 dense/0 的无畸变结果。"""
            dense_root = os.path.join(base_path, "dense", "0")
            dense_sparse = os.path.join(dense_root, "sparse")
            dense_images = os.path.join(dense_root, "images")
            if not (os.path.exists(dense_sparse) and os.path.exists(dense_images)):
                self.log("未找到 dense/0 无畸变结果，无法自动切换。", "error")
                return False
            # 备份现有结构
            backup_dir = os.path.join(base_path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            ts = str(int(time.time()))
            sparse_dir = os.path.join(base_path, "sparse")
            images_dir = os.path.join(base_path, "images")
            if os.path.exists(sparse_dir):
                shutil.move(sparse_dir, os.path.join(backup_dir, f"sparse_orig_{ts}"))
            if os.path.exists(images_dir):
                shutil.move(images_dir, os.path.join(backup_dir, f"images_orig_{ts}"))
            # 建立新结构，指向 dense/0
            os.makedirs(sparse_dir, exist_ok=True)
            try:
                os.symlink(os.path.join(dense_root, "sparse"), os.path.join(sparse_dir, "0"))
            except FileExistsError:
                pass
            try:
                os.symlink(dense_images, images_dir)
            except FileExistsError:
                pass
            self.log(f"已自动切换到无畸变数据: sparse-> {dense_sparse}, images-> {dense_images}")
            return True

        def training_thread():
            try:
                self.is_training = True
                self.set_text_signal.emit(self.training_status_label, "正在训练...")
                self.update_status("正在训练3DGS模型...", 0)
                self.set_enabled_signal.emit(self.start_training_btn, False)
                self.set_enabled_signal.emit(self.stop_training_btn, True)
                
                # 保存训练参数
                training_data_path = data_path
                training_output_path = output_path
                training_iterations = int(self.training_iterations_edit.text())
                
                self.log("开始训练3DGS模型...")
                self.log(f"数据路径: {training_data_path}")
                self.log(f"输出路径: {training_output_path}")
                
                # 构建训练命令
                cmd = [
                    sys.executable, "train.py",
                    "-s", training_data_path,
                    "-m", training_output_path,
                    "--iterations", str(training_iterations),
                    "--resolution", self.training_resolution_combo.currentText(),
                    "--position_lr_init", self.training_lr_edit.text(),
                    "--densify_from_iter", self.densify_from_edit.text(),
                    "--densify_until_iter", self.densify_until_edit.text(),
                    "--test_iterations", self.test_interval_edit.text(),
                    "--save_iterations", self.test_interval_edit.text(),
                    "--quiet"
                ]
                
                self.log(f"训练命令: {' '.join(cmd)}")

                # 最多尝试两次：第一次失败且命中相机模型错误时自动修复并重试一次
                auto_retry_used = False
                while True:
                    # 启动训练进程
                    self.training_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )
                    # 标记是否捕获到相机模型错误
                    cam_model_error = False
                    # 实时读取训练输出
                    for line in iter(self.training_process.stdout.readline, ''):
                        if line.strip():
                            self.log(f"[训练] {line.strip()}")
                            if "Training complete" in line:
                                self.log("训练完成！")
                                break
                            # 识别 3DGS 相机模型断言
                            if "Colmap camera model not handled" in line or "only undistorted datasets" in line:
                                cam_model_error = True
                    # 等待进程结束
                    return_code = self.training_process.wait()
                    if return_code == 0:
                        break
                    # 非零返回码：若命中相机模型错误且尚未重试，执行自动修复并重试
                    if cam_model_error and not auto_retry_used:
                        self.log("检测到相机模型不被支持，尝试自动切换为无畸变数据并重试...")
                        if try_switch_to_undistorted(training_data_path):
                            auto_retry_used = True
                            continue
                        else:
                            pass
                    break
                
                if return_code == 0:
                    self.update_status("训练完成", 100)
                    self.log("3DGS模型训练成功完成！")
                    self.set_text_signal.emit(self.training_status_label, "训练完成")
                    
                    # 自动更新SAGS模型路径和数据路径
                    self.set_text_signal.emit(self.model_path_edit, training_output_path)
                    self.set_text_signal.emit(self.source_path_edit, training_data_path)
                    self.set_text_signal.emit(self.iteration_edit, str(training_iterations))
                    
                    self.log(f"已自动更新SAGS模型路径: {training_output_path}")
                    self.log(f"已自动更新SAGS数据路径: {training_data_path}")
                    
                    # 自动更新SAGA模型路径和数据路径
                    self.set_text_signal.emit(self.saga_model_path_edit, training_output_path)
                    self.set_text_signal.emit(self.saga_source_path_edit, training_data_path)
                    self.set_text_signal.emit(self.saga_iteration_edit, str(training_iterations))
                    
                    self.log(f"已自动更新SAGA模型路径: {training_output_path}")
                    self.log(f"已自动更新SAGA数据路径: {training_data_path}")
                    
                    # 更新系统信息
                    self.update_system_info()
                    
                    # 显示训练完成信息（使用QTimer在主线程中显示）
                    completion_msg = f"""训练完成！

模型路径: {training_output_path}
数据路径: {training_data_path}

是否立即加载训练好的模型？"""
                    
                    def show_training_dialog():
                        reply = QMessageBox.question(self, "训练完成", completion_msg, 
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            self.load_model()
                    QTimer.singleShot(0, show_training_dialog)
                else:
                    self.update_status("训练失败", 0)
                    self.log(f"训练失败，返回码: {return_code}", "error")
                    self.set_text_signal.emit(self.training_status_label, "训练失败")
                    self.show_message("错误", "训练失败，请检查日志", "critical")
                
            except Exception as e:
                self.update_status("训练出错", 0)
                self.log(f"训练过程中发生错误: {str(e)}", "error")
                self.set_text_signal.emit(self.training_status_label, "训练出错")
                self.show_message("错误", f"训练失败: {str(e)}", "critical")
            finally:
                self.is_training = False
                self.set_enabled_signal.emit(self.start_training_btn, True)
                self.set_enabled_signal.emit(self.stop_training_btn, False)
                self.training_process = None
        
        # 在后台线程中运行训练
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """停止训练"""
        if not self.is_training or not self.training_process:
            return
        
        try:
            self.log("正在停止训练...")
            self.training_process.terminate()
            
            # 等待进程结束
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
                self.training_process.wait()
            
            self.log("训练已停止")
            self.set_text_signal.emit(self.training_status_label, "训练已停止")
            
        except Exception as e:
            self.log(f"停止训练时发生错误: {str(e)}")
        finally:
            self.is_training = False
            self.set_enabled_signal.emit(self.start_training_btn, True)
            self.set_enabled_signal.emit(self.stop_training_btn, False)
            self.training_process = None
    
    def validate_sog_training_params(self):
        """验证SOG训练参数"""
        try:
            iterations = int(self.sog_iterations_edit.text())
            if iterations <= 0:
                raise ValueError("迭代次数必须大于0")
            
            # 验证压缩迭代格式
            compress_iter_str = self.sog_compress_iter_edit.text()
            compress_iterations = [int(x.strip()) for x in compress_iter_str.split(',')]
            if not compress_iterations:
                raise ValueError("压缩迭代格式错误")
            
            return True
        except ValueError as e:
            self.show_message("参数错误", str(e), "critical")
            return False
    
    def start_sog_training(self):
        """开始SOG训练"""
        if self.is_sog_training:
            self.show_message("警告", "SOG训练正在进行中", "warning")
            return
        
        # 验证SOG训练参数
        if not self.validate_sog_training_params():
            return
        
        data_path = self.sog_data_path_edit.text()
        original_output_path = self.sog_output_path_edit.text()
        
        if not data_path or not os.path.exists(data_path):
            self.show_message("错误", "请选择有效的训练数据路径", "critical")
            return
        
        # 检查数据集格式
        has_colmap = os.path.exists(os.path.join(data_path, 'sparse'))
        has_blender = os.path.exists(os.path.join(data_path, 'transforms_train.json'))
        
        if not (has_colmap or has_blender):
            self.show_message("错误", "数据集格式不正确。需要包含COLMAP的'sparse'目录或Blender的'transforms_train.json'文件", "critical")
            return
        
        # 创建输出文件夹结构：output/同名文件夹/sog_output
        sags_dir = os.path.dirname(__file__)
        data_folder_name = os.path.basename(data_path.rstrip('/'))
        base_output_dir = os.path.join(sags_dir, "gaussiansplatting", "output")
        
        # 在output路径下新建同名文件夹
        folder_output_path = os.path.join(base_output_dir, data_folder_name)
        os.makedirs(folder_output_path, exist_ok=True)
        
        # 在其中新建sog_output文件夹
        output_path = os.path.join(folder_output_path, "sog_output")
        os.makedirs(output_path, exist_ok=True)
        
        # 更新输出路径显示
        self.set_text_signal.emit(self.sog_output_path_edit, output_path)
        self.log(f"已创建输出文件夹结构: {output_path}")
        
        def sog_training_thread():
            try:
                self.is_sog_training = True
                self.set_text_signal.emit(self.sog_training_status_label, "正在SOG训练...")
                self.update_status("正在训练SOG模型...", 0)
                self.set_enabled_signal.emit(self.start_sog_training_btn, False)
                self.set_enabled_signal.emit(self.stop_sog_training_btn, True)
                
                # 保存训练参数
                sog_data_path = data_path
                sog_output_path = output_path
                sog_iterations = int(self.sog_iterations_edit.text())
                
                self.log("开始SOG训练...")
                self.log(f"数据路径: {sog_data_path}")
                self.log(f"输出路径: {sog_output_path}")
                self.log(f"迭代次数: {sog_iterations}")
                
                # 初始化SOG训练器
                if self.sog_trainer is None:
                    try:
                        self.sog_trainer = SOGTrainer()
                        self.log("SOG训练器初始化成功")
                    except Exception as e:
                        self.log(f"SOG训练器初始化失败: {str(e)}", "error")
                        self.show_message("错误", f"SOG训练器初始化失败: {str(e)}", "critical")
                        return
                
                # 解析压缩迭代
                compress_iter_str = self.sog_compress_iter_edit.text()
                try:
                    compress_iterations = [int(x.strip()) for x in compress_iter_str.split(',')]
                except:
                    compress_iterations = [7000, 10000, 20000, 30000]
                
                # 启动SOG训练
                self.sog_process = self.sog_trainer.train(
                    dataset_path=sog_data_path,
                    output_path=sog_output_path,
                    iterations=sog_iterations,
                    config_name=self.sog_config_combo.currentText(),
                    use_sh=self.sog_use_sh_check.isChecked(),
                    compress_iterations=compress_iterations
                )
                
                self.log("SOG训练进程已启动")
                
                # 实时读取训练输出
                for line in iter(self.sog_process.stdout.readline, ''):
                    if line.strip():
                        self.log(f"[SOG] {line.strip()}")
                        # 检查训练是否完成
                        if "Training complete" in line:
                            self.log("SOG训练完成！")
                            break
                
                # 等待进程结束
                return_code = self.sog_process.wait()
                
                if return_code == 0:
                    self.update_status("SOG训练完成", 100)
                    self.log("SOG模型训练成功完成！")
                    self.set_text_signal.emit(self.sog_training_status_label, "SOG训练完成")
                    
                    # 自动更新SAGS模型路径和数据路径
                    self.set_text_signal.emit(self.model_path_edit, sog_output_path)
                    self.set_text_signal.emit(self.source_path_edit, sog_data_path)
                    self.set_text_signal.emit(self.iteration_edit, str(sog_iterations))
                    
                    self.log(f"已自动更新SAGS模型路径: {sog_output_path}")
                    self.log(f"已自动更新SAGS数据路径: {sog_data_path}")
                    
                    # 自动更新SAGA模型路径和数据路径
                    self.set_text_signal.emit(self.saga_model_path_edit, sog_output_path)
                    self.set_text_signal.emit(self.saga_source_path_edit, sog_data_path)
                    self.set_text_signal.emit(self.saga_iteration_edit, str(sog_iterations))
                    
                    self.log(f"已自动更新SAGA模型路径: {sog_output_path}")
                    self.log(f"已自动更新SAGA数据路径: {sog_data_path}")
                    
                    # 显示训练完成信息（使用QTimer在主线程中显示）
                    completion_msg = f"""SOG训练完成！

模型路径: {sog_output_path}
数据路径: {sog_data_path}

压缩后的模型可以在 {sog_output_path}/compression 目录中找到。

已自动更新SAGS和SAGA的模型加载参数。

是否立即加载训练好的模型？"""
                    
                    def show_sog_completion_dialog():
                        reply = QMessageBox.question(self, "SOG训练完成", completion_msg, 
                                                    QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            self.load_model()
                    QTimer.singleShot(0, show_sog_completion_dialog)
                else:
                    self.update_status("SOG训练失败", 0)
                    self.log(f"SOG训练失败，返回码: {return_code}", "error")
                    self.set_text_signal.emit(self.sog_training_status_label, "SOG训练失败")
                    self.show_message("错误", "SOG训练失败，请检查日志", "critical")
                
            except Exception as e:
                self.update_status("SOG训练出错", 0)
                self.log(f"SOG训练过程中发生错误: {str(e)}", "error")
                self.log(traceback.format_exc(), "error")
                self.set_text_signal.emit(self.sog_training_status_label, "SOG训练出错")
                self.show_message("错误", f"SOG训练失败: {str(e)}", "critical")
            finally:
                self.is_sog_training = False
                self.set_enabled_signal.emit(self.start_sog_training_btn, True)
                self.set_enabled_signal.emit(self.stop_sog_training_btn, False)
                self.sog_process = None
        
        # 在后台线程中运行训练
        self.sog_thread = threading.Thread(target=sog_training_thread, daemon=True)
        self.sog_thread.start()
    
    def stop_sog_training(self):
        """停止SOG训练"""
        if not self.is_sog_training or not self.sog_process:
            return
        
        try:
            self.log("正在停止SOG训练...")
            self.sog_process.terminate()
            
            # 等待进程结束
            try:
                self.sog_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.sog_process.kill()
                self.sog_process.wait()
            
            self.log("SOG训练已停止")
            self.set_text_signal.emit(self.sog_training_status_label, "SOG训练已停止")
            
        except Exception as e:
            self.log(f"停止SOG训练时发生错误: {str(e)}")
        finally:
            self.is_sog_training = False
            self.set_enabled_signal.emit(self.start_sog_training_btn, True)
            self.set_enabled_signal.emit(self.stop_sog_training_btn, False)
            self.sog_process = None
    
    def start_full_workflow(self):
        """一键式处理流程 (COLMAP→3DGS→分割准备)"""
        if self.is_colmap_processing or self.is_training:
            self.show_message("警告", "已有处理任务正在进行中，请等待完成", "warning")
            return
        
        # 验证所有参数
        if not self.validate_colmap_params():
            return
        
        if not self.validate_training_params():
            return
        
        # 确认开始处理
        confirm_msg = """即将开始一键式处理流程：

1. COLMAP处理 (SfM + MVS)
2. 转换为3DGS格式
3. 3DGS模型训练
4. 准备SAGS分割

预计处理时间较长，是否继续？"""
        
        reply = QMessageBox.question(self, "确认一键式处理", confirm_msg, 
                                    QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        def full_workflow_thread():
            try:
                self.log("=== 开始一键式处理流程 ===")
                
                # 步骤1: COLMAP处理
                self.log("步骤 1/4: COLMAP处理")
                self.update_status("步骤 1/4: COLMAP处理", 0)
                
                colmap_input_path = self.colmap_input_path_edit.text()
                # 自动设置COLMAP输出路径
                input_folder_name = os.path.basename(colmap_input_path.rstrip('/'))
                sags_dir = os.path.dirname(__file__)
                colmap_output_path = os.path.join(sags_dir, "gaussiansplatting", "input", input_folder_name)
                camera_model = self.colmap_camera_model_combo.currentText()
                single_camera = self.colmap_single_camera_check.isChecked()
                
                # 创建COLMAP处理器
                self.colmap_processor = COLMAPProcessor()
                
                def colmap_progress_callback(message, progress=None):
                    if progress is not None:
                        self.update_status(f"COLMAP: {message}", progress * 0.25)  # COLMAP占25%
                        self.log(f"[COLMAP {progress:.0f}%] {message}")
                    else:
                        self.log(f"[COLMAP] {message}")
                
                self.colmap_processor.set_progress_callback(colmap_progress_callback)
                
                # 使用COLMAP自动重建
                colmap_success = self.colmap_processor.auto_reconstruction(
                    image_dir=colmap_input_path,
                    workspace_path=colmap_output_path,
                    camera_model=camera_model,
                    single_camera=single_camera,
                    quality=self.colmap_quality_combo.currentText(),
                    data_type=self.colmap_data_type_combo.currentText(),
                    mapper_type=self.colmap_mapper_type_combo.currentText(),
                    num_threads=self.colmap_num_threads_edit.text(),
                    sparse_model=self.colmap_sparse_model_check.isChecked(),
                    dense_model=self.colmap_dense_model_check.isChecked(),
                    use_gpu=self.colmap_use_gpu_check.isChecked(),
                    vocab_tree=self.colmap_vocab_tree_edit.text() if self.colmap_vocab_tree_edit.text() and os.path.exists(self.colmap_vocab_tree_edit.text()) else None
                )
                
                if not colmap_success:
                    self.log("COLMAP处理失败，终止流程", "error")
                    return
                
                self.log("COLMAP处理完成")
                
                # 步骤2: 自动路径管理
                self.log("步骤 2/4: 自动路径管理")
                self.update_status("步骤 2/4: 自动路径管理", 25)
                
                # 设置训练数据路径（COLMAP输出路径）
                training_data_path = colmap_output_path
                self.set_text_signal.emit(self.training_data_path_edit, training_data_path)
                
                # 设置训练输出路径
                training_output_path = os.path.join(sags_dir, "gaussiansplatting", "output", input_folder_name)
                self.set_text_signal.emit(self.training_output_path_edit, training_output_path)
                
                # 设置模型路径
                self.set_text_signal.emit(self.model_path_edit, training_output_path)
                self.set_text_signal.emit(self.source_path_edit, training_data_path)
                
                self.log(f"训练数据路径: {training_data_path}")
                self.log(f"训练输出路径: {training_output_path}")
                
                # 步骤3: 3DGS训练
                self.log("步骤 3/4: 3DGS模型训练")
                self.update_status("步骤 3/4: 3DGS模型训练", 50)
                
                training_iterations = int(self.training_iterations_edit.text())
                
                # 构建训练命令
                cmd = [
                    sys.executable, "train.py",
                    "-s", training_data_path,
                    "-m", training_output_path,
                    "--iterations", str(training_iterations),
                    "--resolution", self.training_resolution_combo.currentText(),
                    "--position_lr_init", self.training_lr_edit.text(),
                    "--densify_from_iter", self.densify_from_edit.text(),
                    "--densify_until_iter", self.densify_until_edit.text(),
                    "--test_iterations", self.test_interval_edit.text(),
                    "--save_iterations", self.test_interval_edit.text(),
                    "--quiet"
                ]
                
                self.log(f"训练命令: {' '.join(cmd)}")
                
                # 启动训练进程
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时读取训练输出
                for line in iter(self.training_process.stdout.readline, ''):
                    if line.strip():
                        self.log(f"[训练] {line.strip()}")
                        # 更新进度 (50% + 训练进度 * 0.4)
                        if "Training complete" in line:
                            self.log("训练完成！")
                            break
                
                # 等待进程结束
                return_code = self.training_process.wait()
                
                if return_code != 0:
                    self.log("3DGS训练失败，终止流程", "error")
                    return
                
                self.log("3DGS训练完成")
                
                # 步骤4: 准备SAGS分割
                self.log("步骤 4/4: 准备SAGS分割")
                self.update_status("步骤 4/4: 准备SAGS分割", 90)
                
                # 自动更新迭代次数
                self.set_text_signal.emit(self.iteration_edit, str(training_iterations))
                
                self.log("=== 一键式处理流程完成 ===")
                self.update_status("一键式处理完成", 100)
                
                # 显示完成信息
                completion_msg = f"""一键式处理流程完成！

COLMAP输出: {colmap_output_path}
训练数据: {training_data_path}
训练输出: {training_output_path}

现在可以开始SAGS分割操作。"""
                
                self.show_message("处理完成", completion_msg, "info")
                
                # 询问是否立即加载模型（使用QTimer在主线程中显示）
                def show_load_dialog():
                    reply = QMessageBox.question(self, "加载模型", "是否立即加载训练好的模型？", 
                                                QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        self.load_model()
                QTimer.singleShot(0, show_load_dialog)
                
            except Exception as e:
                self.log(f"一键式处理流程出错: {str(e)}", "error")
                self.show_message("错误", f"一键式处理失败: {str(e)}", "critical")
            finally:
                # 清理状态
                self.colmap_processor = None
                self.training_process = None
                self.is_colmap_processing = False
                self.is_training = False
        
        # 在后台线程中运行完整流程
        workflow_thread = threading.Thread(target=full_workflow_thread, daemon=True)
        workflow_thread.start()
    
    def start_full_workflow_with_sags(self):
        """一键式处理流程 (COLMAP→3DGS→完整分割)"""
        if self.is_colmap_processing or self.is_training:
            self.show_message("警告", "已有处理任务正在进行中，请等待完成", "warning")
            return
        
        # 验证所有参数
        if not self.validate_colmap_params():
            return
        
        if not self.validate_training_params():
            return
        
        # 检查是否启用了预处理
        if not self.enable_preprocess_check.isChecked():
            self.show_message("警告", "请先启用分割预处理并设置提示参数", "warning")
            return
        
        # 验证预处理参数
        if not self.validate_preprocess_params():
            return
        
        # 确认开始处理
        confirm_msg = """即将开始一键式处理流程：

1. COLMAP处理 (SfM + MVS)
2. 转换为3DGS格式
3. 3DGS模型训练
4. SAGS分割处理
5. 保存分割结果

预计处理时间较长，是否继续？"""
        
        reply = QMessageBox.question(self, "确认一键式处理", confirm_msg, 
                                    QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        def full_workflow_sags_thread():
            try:
                self.log("=== 开始一键式处理流程 (含完整分割) ===")
                
                # 步骤1: COLMAP处理
                self.log("步骤 1/5: COLMAP处理")
                self.update_status("步骤 1/5: COLMAP处理", 0)
                
                colmap_input_path = self.colmap_input_path_edit.text()
                # 自动设置COLMAP输出路径
                input_folder_name = os.path.basename(colmap_input_path.rstrip('/'))
                sags_dir = os.path.dirname(__file__)
                colmap_output_path = os.path.join(sags_dir, "gaussiansplatting", "input", input_folder_name)
                camera_model = self.colmap_camera_model_combo.currentText()
                single_camera = self.colmap_single_camera_check.isChecked()
                
                # 创建COLMAP处理器
                self.colmap_processor = COLMAPProcessor()
                
                def colmap_progress_callback(message, progress=None):
                    if progress is not None:
                        self.update_status(f"COLMAP: {message}", progress * 0.2)  # COLMAP占20%
                        self.log(f"[COLMAP {progress:.0f}%] {message}")
                    else:
                        self.log(f"[COLMAP] {message}")
                
                self.colmap_processor.set_progress_callback(colmap_progress_callback)
                
                # 使用COLMAP自动重建
                colmap_success = self.colmap_processor.auto_reconstruction(
                    image_dir=colmap_input_path,
                    workspace_path=colmap_output_path,
                    camera_model=camera_model,
                    single_camera=single_camera,
                    quality=self.colmap_quality_combo.currentText(),
                    data_type=self.colmap_data_type_combo.currentText(),
                    mapper_type=self.colmap_mapper_type_combo.currentText(),
                    num_threads=self.colmap_num_threads_edit.text(),
                    sparse_model=self.colmap_sparse_model_check.isChecked(),
                    dense_model=self.colmap_dense_model_check.isChecked(),
                    use_gpu=self.colmap_use_gpu_check.isChecked(),
                    vocab_tree=self.colmap_vocab_tree_edit.text() if self.colmap_vocab_tree_edit.text() and os.path.exists(self.colmap_vocab_tree_edit.text()) else None
                )
                
                if not colmap_success:
                    self.log("COLMAP处理失败，终止流程", "error")
                    return
                
                self.log("COLMAP处理完成")
                
                # 步骤2: 自动路径管理
                self.log("步骤 2/5: 自动路径管理")
                self.update_status("步骤 2/5: 自动路径管理", 20)
                
                # 设置训练数据路径（COLMAP输出路径）
                training_data_path = colmap_output_path
                self.set_text_signal.emit(self.training_data_path_edit, training_data_path)
                
                # 设置训练输出路径
                training_output_path = os.path.join(sags_dir, "gaussiansplatting", "output", input_folder_name)
                self.set_text_signal.emit(self.training_output_path_edit, training_output_path)
                
                # 设置模型路径
                self.set_text_signal.emit(self.model_path_edit, training_output_path)
                self.set_text_signal.emit(self.source_path_edit, training_data_path)
                
                self.log(f"训练数据路径: {training_data_path}")
                self.log(f"训练输出路径: {training_output_path}")
                
                # 步骤3: 3DGS训练
                self.log("步骤 3/5: 3DGS模型训练")
                self.update_status("步骤 3/5: 3DGS模型训练", 40)
                
                training_iterations = int(self.training_iterations_edit.text())
                
                # 构建训练命令
                cmd = [
                    sys.executable, "train.py",
                    "-s", training_data_path,
                    "-m", training_output_path,
                    "--iterations", str(training_iterations),
                    "--resolution", self.training_resolution_combo.currentText(),
                    "--position_lr_init", self.training_lr_edit.text(),
                    "--densify_from_iter", self.densify_from_edit.text(),
                    "--densify_until_iter", self.densify_until_edit.text(),
                    "--test_iterations", self.test_interval_edit.text(),
                    "--save_iterations", self.test_interval_edit.text(),
                    "--quiet"
                ]
                
                self.log(f"训练命令: {' '.join(cmd)}")
                
                # 启动训练进程
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 实时读取训练输出
                for line in iter(self.training_process.stdout.readline, ''):
                    if line.strip():
                        self.log(f"[训练] {line.strip()}")
                        # 更新进度 (40% + 训练进度 * 0.3)
                        if "Training complete" in line:
                            self.log("训练完成！")
                            break
                
                # 等待进程结束
                return_code = self.training_process.wait()
                
                if return_code != 0:
                    self.log("3DGS训练失败，终止流程", "error")
                    return
                
                self.log("3DGS训练完成")
                
                # 步骤4: 加载模型并准备SAGS分割
                self.log("步骤 4/5: 准备SAGS分割")
                self.update_status("步骤 4/5: 准备SAGS分割", 70)
                
                # 自动更新迭代次数
                self.set_text_signal.emit(self.iteration_edit, str(training_iterations))
                
                # 直接加载训练好的模型（同步方式）
                self.log("正在加载训练好的模型...")
                if not self.load_model_sync():
                    self.log("模型加载失败，终止流程", "error")
                    return
                
                # 步骤5: 执行SAGS分割
                self.log("步骤 5/5: 执行SAGS分割")
                self.update_status("步骤 5/5: 执行SAGS分割", 80)
                
                # 使用预处理参数执行分割
                self.run_sags_with_preprocess()
                
                self.log("=== 一键式处理流程完成 ===")
                self.update_status("一键式处理完成", 100)
                
                # 显示完成信息
                completion_msg = f"""一键式处理流程完成！

COLMAP输出: {colmap_output_path}
训练数据: {training_data_path}
训练输出: {training_output_path}
分割结果: {training_output_path}/point_cloud/iteration_666/

模型已自动加载，可以查看分割结果。"""
                
                self.show_message("处理完成", completion_msg, "info")
                
            except Exception as e:
                self.log(f"一键式处理流程出错: {str(e)}", "error")
                self.show_message("错误", f"一键式处理失败: {str(e)}", "critical")
            finally:
                # 清理状态
                self.colmap_processor = None
                self.training_process = None
                self.is_colmap_processing = False
                self.is_training = False
        
        # 在后台线程中运行完整流程
        workflow_thread = threading.Thread(target=full_workflow_sags_thread, daemon=True)
        workflow_thread.start()
    
    # SAGA相关方法（如果SAGA可用）
    
    def saga_extract_masks(self):
        """提取SAM masks"""
        if not SAGA_AVAILABLE or not self.saga_module:
            self.log("SAGA模块不可用", "error")
            return
        
        if not self.saga_module.source_path:
            self.log("错误: 请先加载模型", "error")
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        # 获取SAM模型路径（SAGA只支持标准SAM架构，不支持FastSAM）
        sam_model_type = self.saga_sam_model_combo.currentText()
        sam_model_paths = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth', 
            'vit_b': 'sam_vit_b_01ec64.pth'
        }
        sam_ckpt_filename = sam_model_paths.get(sam_model_type, 'sam_vit_h_4b8939.pth')
        sam_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
        
        if not os.path.exists(sam_path):
            self.log(f"错误: SAM模型文件不存在: {sam_path}", "error")
            self.show_message("错误", f"SAM模型文件不存在: {sam_path}", "critical")
            return
        
        def extract_thread():
            try:
                downsample = int(self.saga_downsample_edit.text())
                # 从SAM模型类型自动推断架构（SAGA只支持标准SAM架构）
                sam_model_type = self.saga_sam_model_combo.currentText()
                # 确保架构与模型类型一致（SAGA不支持FastSAM，只支持vit_h/vit_l/vit_b）
                if sam_model_type in ['vit_h', 'vit_l', 'vit_b']:
                    sam_arch = sam_model_type
                else:
                    sam_arch = 'vit_h'  # 默认值
                downsample_type = self.saga_downsample_type_combo.currentText()
                max_long_side = int(self.saga_max_long_side_combo.currentText())
                
                # 清理GPU内存
                MemoryManager.clear_gpu_memory()
                
                def progress_callback(current, total, message=""):
                    if total > 0:
                        progress = int(current / total * 100)
                        self.set_text_signal.emit(self.saga_status_label, f"提取SAM masks: {message} ({progress}%)")
                    self.update_status(f"提取SAM masks: {message}", progress if total > 0 else None)
                
                self.log(f"开始提取SAM masks...", "info")
                self.set_text_signal.emit(self.saga_status_label, "正在提取SAM masks...")
                
                success = self.saga_module.extract_sam_masks(
                    sam_checkpoint_path=sam_path,
                    downsample=downsample,
                    sam_arch=sam_arch,
                    downsample_type=downsample_type,
                    max_long_side=max_long_side,
                    callback=progress_callback
                )
                
                if success:
                    self.log("SAM masks提取完成", "info")
                    self.set_text_signal.emit(self.saga_status_label, "SAM masks提取完成")
                    self.show_message("成功", "SAM masks提取完成", "info")
                else:
                    self.log("SAM masks提取失败", "error")
                    self.set_text_signal.emit(self.saga_status_label, "SAM masks提取失败")
                    self.show_message("错误", "SAM masks提取失败", "critical")
                    
            except Exception as e:
                self.log(f"提取SAM masks失败: {str(e)}", "error")
                self.set_text_signal.emit(self.saga_status_label, f"错误: {str(e)}")
                self.show_message("错误", f"提取失败: {str(e)}", "critical")
                traceback.print_exc()
        
        threading.Thread(target=extract_thread, daemon=True).start()
    
    def saga_get_scales(self):
        """计算mask scales"""
        if not SAGA_AVAILABLE or not self.saga_module:
            self.log("SAGA模块不可用", "error")
            return
        
        if not self.saga_module.source_path:
            self.log("错误: 请先加载模型", "error")
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        def scales_thread():
            try:
                def progress_callback(current, total, message=""):
                    if total > 0:
                        progress = int(current / total * 100)
                        self.set_text_signal.emit(self.saga_status_label, f"计算mask scales: {message} ({progress}%)")
                    self.update_status(f"计算mask scales: {message}", progress if total > 0 else None)
                
                self.log(f"开始计算mask scales...", "info")
                self.set_text_signal.emit(self.saga_status_label, "正在计算mask scales...")
                
                success = self.saga_module.get_mask_scales(callback=progress_callback)
                
                if success:
                    self.log("Mask scales计算完成", "info")
                    self.set_text_signal.emit(self.saga_status_label, "Mask scales计算完成")
                    self.show_message("成功", "Mask scales计算完成", "info")
                else:
                    self.log("Mask scales计算失败", "error")
                    self.set_text_signal.emit(self.saga_status_label, "Mask scales计算失败")
                    self.show_message("错误", "Mask scales计算失败", "critical")
                    
            except Exception as e:
                self.log(f"计算mask scales失败: {str(e)}", "error")
                self.set_text_signal.emit(self.saga_status_label, f"错误: {str(e)}")
                self.show_message("错误", f"计算失败: {str(e)}", "critical")
                traceback.print_exc()
        
        threading.Thread(target=scales_thread, daemon=True).start()
    
    def saga_prepare_all_data(self):
        """一键准备所有数据"""
        if not SAGA_AVAILABLE or not self.saga_module:
            self.log("SAGA模块不可用", "error")
            return
        
        if not self.saga_module.source_path:
            self.log("错误: 请先加载模型", "error")
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        # 获取SAM模型路径（SAGA只支持标准SAM架构，不支持FastSAM）
        sam_model_type = self.saga_sam_model_combo.currentText()
        sam_model_paths = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth', 
            'vit_b': 'sam_vit_b_01ec64.pth'
        }
        sam_ckpt_filename = sam_model_paths.get(sam_model_type, 'sam_vit_h_4b8939.pth')
        sam_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
        
        if not os.path.exists(sam_path):
            self.log(f"错误: SAM模型文件不存在: {sam_path}", "error")
            self.show_message("错误", f"SAM模型文件不存在: {sam_path}", "critical")
            return
        
        def prepare_thread():
            try:
                downsample = int(self.saga_downsample_edit.text())
                # 从SAM模型类型自动推断架构（SAGA只支持标准SAM架构）
                sam_model_type = self.saga_sam_model_combo.currentText()
                # 确保架构与模型类型一致（SAGA不支持FastSAM，只支持vit_h/vit_l/vit_b）
                if sam_model_type in ['vit_h', 'vit_l', 'vit_b']:
                    sam_arch = sam_model_type
                else:
                    sam_arch = 'vit_h'  # 默认值
                downsample_type = self.saga_downsample_type_combo.currentText()
                max_long_side = int(self.saga_max_long_side_combo.currentText())
                
                # 清理GPU内存
                MemoryManager.clear_gpu_memory()
                
                def progress_callback(current, total, message=""):
                    if total > 0:
                        progress = int(current / total * 100)
                        self.set_text_signal.emit(self.saga_status_label, f"准备数据: {message} ({progress}%)")
                    self.update_status(f"准备数据: {message}", progress if total > 0 else None)
                
                self.log(f"开始一键准备所有数据...", "info")
                self.set_text_signal.emit(self.saga_status_label, "正在准备数据...")
                
                success = self.saga_module.prepare_all_data(
                    sam_checkpoint_path=sam_path,
                    downsample=downsample,
                    sam_arch=sam_arch,
                    downsample_type=downsample_type,
                    max_long_side=max_long_side,
                    callback=progress_callback
                )
                
                if success:
                    self.log("所有数据准备完成！", "info")
                    self.set_text_signal.emit(self.saga_status_label, "数据准备完成")
                    self.show_message("成功", "所有数据准备完成！", "info")
                else:
                    self.log("数据准备失败", "error")
                    self.set_text_signal.emit(self.saga_status_label, "数据准备失败")
                    self.show_message("错误", "数据准备失败", "critical")
                    
            except Exception as e:
                self.log(f"数据准备失败: {str(e)}", "error")
                self.set_text_signal.emit(self.saga_status_label, f"错误: {str(e)}")
                self.show_message("错误", f"数据准备失败: {str(e)}", "critical")
                traceback.print_exc()
        
        threading.Thread(target=prepare_thread, daemon=True).start()
    
    def saga_check_feature_model(self):
        """检查特征模型是否存在"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.model_loaded or not self.saga_module:
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        try:
            exists = self.saga_module.check_feature_model_exists()
            if exists:
                self.set_text_signal.emit(self.saga_status_label, "特征模型存在")
                self.log("特征模型检查: 存在", "info")
                self.show_message("检查结果", "特征模型存在", "info")
            else:
                self.set_text_signal.emit(self.saga_status_label, "特征模型不存在")
                self.log("特征模型检查: 不存在", "warning")
                self.show_message("检查结果", "特征模型不存在，请先训练对比特征", "warning")
        except Exception as e:
            self.log(f"检查特征模型失败: {str(e)}", "error")
            self.show_message("错误", f"检查失败: {str(e)}", "critical")
    
    def saga_load_feature_model(self):
        """加载特征模型"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.model_loaded or not self.saga_module:
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        def load_thread():
            try:
                self.update_status("正在加载特征模型...", 0)
                self.log("开始加载特征模型...")
                
                success = self.saga_module.load_feature_model()
                
                if success:
                    self.saga_feature_loaded = True
                    self.set_text_signal.emit(self.saga_status_label, "特征模型已加载")
                    self.update_status("特征模型加载完成", 100)
                    self.log("特征模型加载成功", "info")
                    self.show_message("成功", "特征模型加载成功", "info")
                    self.saga_update_visualization()
                else:
                    self.set_text_signal.emit(self.saga_status_label, "特征模型加载失败")
                    self.update_status("特征模型加载失败", 0)
                    self.log("特征模型加载失败", "error")
                    self.show_message("错误", "特征模型加载失败", "critical")
            except Exception as e:
                self.update_status("特征模型加载失败", 0)
                self.log(f"加载特征模型失败: {str(e)}", "error")
                traceback.print_exc()
                self.show_message("错误", f"加载失败: {str(e)}", "critical")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def saga_train_features(self):
        """训练对比特征"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.model_loaded or not self.saga_module:
            self.show_message("错误", "请先加载模型", "critical")
            return
        
        def train_thread():
            try:
                iterations = int(self.saga_train_iterations_edit.text())
                num_rays = int(self.saga_num_rays_edit.text())
                smooth_k = int(self.saga_smooth_k_edit.text())
                feature_lr = float(self.saga_feature_lr_edit.text())
                
                self.update_status("正在训练对比特征...", 0)
                self.log(f"开始训练对比特征: 迭代={iterations}, 光线数={num_rays}")
                
                # 创建进度回调函数，将训练进度实时输出到GUI
                # 使用QMetaObject.invokeMethod确保线程安全
                def progress_callback(iteration, total, loss_info):
                    """
                    训练进度回调
                    Args:
                        iteration: 当前迭代
                        total: 总迭代数
                        loss_info: 损失信息字典 {'loss': ..., 'rfn': ..., 'pos_cos': ..., 'neg_cos': ...}
                    """
                    try:
                        progress = int((iteration / total) * 100)
                        self.update_status(f"训练进度: {iteration}/{total}", progress)
                        
                        # 每100次迭代输出一次详细信息到日志
                        if iteration % 100 == 0 or iteration == total:
                            loss_str = f"Loss={loss_info.get('loss', 'N/A'):.4f}" if isinstance(loss_info.get('loss'), (int, float)) else f"Loss={loss_info.get('loss', 'N/A')}"
                            rfn_str = f"RFN={loss_info.get('rfn', 'N/A'):.4f}" if isinstance(loss_info.get('rfn'), (int, float)) else f"RFN={loss_info.get('rfn', 'N/A')}"
                            pos_str = f"Pos={loss_info.get('pos_cos', 'N/A'):.4f}" if isinstance(loss_info.get('pos_cos'), (int, float)) else f"Pos={loss_info.get('pos_cos', 'N/A')}"
                            neg_str = f"Neg={loss_info.get('neg_cos', 'N/A'):.4f}" if isinstance(loss_info.get('neg_cos'), (int, float)) else f"Neg={loss_info.get('neg_cos', 'N/A')}"
                            
                            self.log(f"[训练] Iter {iteration}/{total}: {loss_str}, {rfn_str}, {pos_str}, {neg_str}", "info")
                            
                            # 检测NaN并警告
                            if 'nan' in loss_str.lower() or 'nan' in rfn_str.lower():
                                self.log(f"⚠️ 警告: 检测到NaN值！训练可能不稳定", "warning")
                    except Exception as e:
                        # 捕获回调中的任何错误，避免影响训练
                        pass
                
                def log_callback(message, level='info'):
                    """
                    日志回调函数
                    Args:
                        message: 日志消息
                        level: 日志级别 ('info', 'warning', 'error')
                    """
                    try:
                        if level == 'warning':
                            self.log(f"[训练] {message}", "warning")
                        elif level == 'error':
                            self.log(f"[训练] {message}", "error")
                        else:
                            self.log(f"[训练] {message}", "info")
                    except Exception as e:
                        # 捕获回调中的任何错误，避免影响训练
                        pass
                
                success = self.saga_module.train_contrastive_features(
                    iterations=iterations,
                    num_sampled_rays=num_rays,
                    smooth_K=smooth_k,
                    feature_lr=feature_lr,
                    callback=progress_callback,
                    log_callback=log_callback
                )
                
                if success:
                    self.update_status("训练完成", 100)
                    self.log("对比特征训练完成", "info")
                    self.show_message("成功", "对比特征训练完成", "info")
                    self.set_text_signal.emit(self.saga_status_label, "特征模型已训练，请加载")
                else:
                    self.update_status("训练失败", 0)
                    self.log("对比特征训练失败", "error")
                    self.show_message("错误", "训练失败", "critical")
            except Exception as e:
                self.update_status("训练失败", 0)
                self.log(f"训练对比特征失败: {str(e)}", "error")
                traceback.print_exc()
                self.show_message("错误", f"训练失败: {str(e)}", "critical")
        
        reply = QMessageBox.question(self, "确认", "训练对比特征可能需要较长时间，是否继续？", 
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            threading.Thread(target=train_thread, daemon=True).start()
    
    def on_saga_point_type_change(self):
        """SAGA点提示类型改变"""
        if not SAGA_AVAILABLE:
            return
        point_type = self.saga_point_type_combo.currentText()
        self.saga_current_point_type = 1 if point_type == "正点" else 0
        self.log(f"SAGA点提示类型: {point_type}")
    
    def on_saga_canvas_click(self, x, y):
        """SAGA画布点击事件"""
        if not SAGA_AVAILABLE:
            return
        if not self.model_loaded or not self.saga_module or not self.saga_feature_loaded:
            self.log("请先加载模型和特征模型", "warning")
            return
        
        try:
            canvas = self.saga_canvas
            # 将窗口坐标转换为图像坐标
            img_x = int((x - canvas.offset_x) / canvas.zoom)
            img_y = int((y - canvas.offset_y) / canvas.zoom)
            
            # 确保在图像范围内
            if canvas.image is not None:
                img_h, img_w = canvas.image.shape[:2]
                img_x = max(0, min(img_w - 1, img_x))
                img_y = max(0, min(img_h - 1, img_y))
            
            # 添加点提示
            if self.saga_module:
                self.saga_module.add_point_prompt(img_x, img_y, self.saga_current_point_type)
            
            # 在画布上添加点显示
            canvas.add_point(x, y, self.saga_current_point_type)
            self.saga_point_prompts = canvas.click_points
            
            # 更新可视化
            self.saga_update_visualization()
            
            label = "正点" if self.saga_current_point_type == 1 else "负点"
            self.log(f"添加SAGA{label}: ({img_x}, {img_y})")
        except Exception as e:
            self.log(f"SAGA点击处理失败: {str(e)}", "error")
    
    def saga_clear_points(self):
        """清空SAGA点提示"""
        if not SAGA_AVAILABLE or not self.saga_module:
            return
        if self.saga_module:
            self.saga_module.clear_point_prompts()
        self.saga_point_prompts = []
        self.saga_canvas.clear_points()
        self.log("已清空SAGA点提示")
        self.saga_update_visualization()
    
    def saga_run_segmentation(self):
        """执行SAGA 3D分割"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.model_loaded or not self.saga_module or not self.saga_feature_loaded:
            self.show_message("错误", "请先加载模型和特征模型", "critical")
            return
        
        if not self.saga_point_prompts:
            self.show_message("错误", "请先添加点提示", "critical")
            return
        
        def seg_thread():
            try:
                score_thresh = float(self.saga_score_thresh_edit.text())
                scale = float(self.saga_scale_edit.text())
                
                self.update_status("正在执行SAGA分割...", 0)
                self.log(f"开始SAGA分割: 阈值={score_thresh}, 尺度={scale}")
                self.log(f"提示：如果选中0个点，请查看终端输出的调试信息，并降低阈值", "info")
                
                mask = self.saga_module.segment_from_points(
                    self.current_view_idx,
                    score_thresh=score_thresh,
                    scale=scale
                )
                
                # 确保mask是布尔类型tensor，并保存到final_mask以供GUI使用
                if isinstance(mask, torch.Tensor):
                    # 确保是布尔类型
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    # 保存到final_mask，与其他分割方法保持一致
                    self.final_mask = mask.cpu() if mask.device.type == 'cuda' else mask.clone()
                else:
                    # 如果不是tensor，转换为tensor
                    self.final_mask = torch.tensor(mask, dtype=torch.bool)
                
                # 保存当前迭代信息，供SIBR查看器使用
                if self.saga_module and hasattr(self.saga_module, 'iteration'):
                    self.segmentation_iteration = self.saga_module.iteration
                else:
                    # 尝试从模型路径获取迭代信息
                    self.segmentation_iteration = None
                
                self.update_status("分割完成", 100)
                mask_sum = mask.sum().item() if hasattr(mask, 'sum') else (mask.sum() if hasattr(mask, '__len__') else 0)
                
                # 根据选中点数给出不同的反馈
                if mask_sum == 0:
                    self.log(f"SAGA分割完成，但选中了 0 个高斯点", "warning")
                    self.log(f"建议：请查看终端输出的相似度统计信息", "warning")
                    self.log(f"提示：尝试降低阈值（建议范围0.1-0.3）或调整点提示位置", "info")
                    self.show_message("警告", "分割完成但选中了0个点\n请降低阈值或调整点提示\n详细信息请查看终端输出", "warning")
                elif mask_sum < 100:
                    self.log(f"SAGA分割完成，选中 {mask_sum} 个高斯点（较少）", "warning")
                    self.log(f"提示：如果结果不理想，可以尝试降低阈值", "info")
                    self.show_message("成功", f"分割完成，选中 {mask_sum} 个高斯点\n（点数较少，可能需要调整阈值）", "info")
                else:
                    self.log(f"SAGA分割完成，选中 {mask_sum} 个高斯点", "info")
                    self.show_message("成功", f"分割完成，选中 {mask_sum} 个高斯点", "info")
                
                # 更新可视化
                self.saga_update_visualization()
                
                # 更新3D渲染器显示分割结果
                self.update_3d_viewer_with_segmentation()
                
            except Exception as e:
                self.update_status("分割失败", 0)
                self.log(f"SAGA分割失败: {str(e)}", "error")
                traceback.print_exc()
                self.show_message("错误", f"分割失败: {str(e)}", "critical")
        
        threading.Thread(target=seg_thread, daemon=True).start()
    
    def saga_undo(self):
        """撤销SAGA分割"""
        if not SAGA_AVAILABLE or not self.saga_module:
            return
        if self.saga_module.undo_segmentation():
            # 同步更新final_mask
            if self.saga_module.current_mask is not None:
                mask = self.saga_module.current_mask
                if isinstance(mask, torch.Tensor):
                    if mask.dtype != torch.bool:
                        mask = mask.bool()
                    self.final_mask = mask.cpu() if mask.device.type == 'cuda' else mask.clone()
                else:
                    self.final_mask = torch.tensor(mask, dtype=torch.bool)
            else:
                self.final_mask = None
            self.log("已撤销SAGA分割")
            self.saga_update_visualization()
        else:
            self.log("没有可撤销的分割", "warning")
    
    def saga_clear_segmentation(self):
        """清空SAGA分割"""
        if not SAGA_AVAILABLE or not self.saga_module:
            return
        self.saga_module.clear_segmentation()
        # 同步清空final_mask
        self.final_mask = None
        self.saga_point_prompts = []
        self.saga_canvas.clear_points()
        self.log("已清空SAGA分割")
        self.saga_update_visualization()
    
    def saga_save_segmentation(self):
        """保存SAGA分割结果"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.saga_module:
            self.show_message("错误", "SAGA模块未初始化", "critical")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存SAGA分割结果", "", 
            "PyTorch Files (*.pt);;NumPy Files (*.npy);;All Files (*.*)"
        )
        
        if save_path:
            try:
                self.saga_module.save_segmentation(save_path)
                self.log(f"SAGA分割已保存: {save_path}", "info")
                self.show_message("成功", "分割结果已保存", "info")
            except Exception as e:
                self.log(f"保存失败: {str(e)}", "error")
                self.show_message("错误", f"保存失败: {str(e)}", "critical")
    
    def saga_run_clustering(self):
        """执行SAGA 3D聚类"""
        if not SAGA_AVAILABLE:
            self.show_message("错误", "SAGA模块不可用", "critical")
            return
        if not self.model_loaded or not self.saga_module or not self.saga_feature_loaded:
            self.show_message("错误", "请先加载模型和特征模型", "critical")
            return
        
        def cluster_thread():
            try:
                min_cluster = int(self.saga_min_cluster_edit.text())
                min_samples = int(self.saga_min_samples_edit.text())
                
                self.update_status("正在执行3D聚类...", 0)
                self.log(f"开始SAGA 3D聚类: 最小聚类={min_cluster}, 最小样本={min_samples}")
                
                labels = self.saga_module.cluster_3d(
                    min_cluster_size=min_cluster,
                    min_samples=min_samples
                )
                
                import torch
                n_clusters = len(torch.unique(labels[labels >= 0]))
                self.update_status("聚类完成", 100)
                self.log(f"SAGA 3D聚类完成，找到 {n_clusters} 个聚类", "info")
                
                # 更新可视化
                self.saga_update_visualization()
                
                self.show_message("成功", f"聚类完成，找到 {n_clusters} 个聚类", "info")
            except Exception as e:
                self.update_status("聚类失败", 0)
                self.log(f"SAGA聚类失败: {str(e)}", "error")
                traceback.print_exc()
                self.show_message("错误", f"聚类失败: {str(e)}", "critical")
        
        threading.Thread(target=cluster_thread, daemon=True).start()
        
    def saga_update_visualization(self):
        """更新SAGA可视化"""
        if not SAGA_AVAILABLE:
            return
        if not self.model_loaded or not self.saga_module:
            return
        
        try:
            viz_mode_button = self.saga_viz_mode_group.checkedButton()
            if not viz_mode_button:
                viz_mode = "RGB"
            else:
                viz_mode_index = self.saga_viz_mode_group.id(viz_mode_button)
                viz_modes = ["RGB", "PCA特征", "相似度图", "3D聚类", "分割结果"]
                viz_mode = viz_modes[viz_mode_index] if viz_mode_index < len(viz_modes) else "RGB"
            
            view_idx = self.current_view_idx
            image = None
            fallback_to_rgb = False
            
            if viz_mode == "RGB":
                image = self.saga_module.render_rgb(view_idx)
            elif viz_mode == "PCA特征":
                if not self.saga_feature_loaded:
                    self.log("请先加载特征模型，显示RGB图像", "warning")
                    fallback_to_rgb = True
                else:
                    image = self.saga_module.render_pca_features(view_idx)
            elif viz_mode == "相似度图":
                if not self.saga_feature_loaded or not self.saga_point_prompts:
                    self.log("请先加载特征模型并添加点提示，显示RGB图像", "warning")
                    fallback_to_rgb = True
                else:
                    score_thresh = float(self.saga_score_thresh_edit.text())
                    scale = float(self.saga_scale_edit.text())
                    image = self.saga_module.render_similarity_map(view_idx, score_thresh, scale)
            elif viz_mode == "3D聚类":
                if not hasattr(self.saga_module, 'cluster_labels') or self.saga_module.cluster_labels is None:
                    self.log("请先执行3D聚类，显示RGB图像", "warning")
                    fallback_to_rgb = True
                else:
                    image = self.saga_module.render_cluster(view_idx)
            elif viz_mode == "分割结果":
                if self.saga_module.current_mask is None:
                    self.log("请先执行分割，显示RGB图像", "warning")
                    fallback_to_rgb = True
                else:
                    image = self.saga_module.render_segmentation(view_idx)
            
            # 如果渲染失败或需要fallback，尝试渲染RGB图像
            if image is None or fallback_to_rgb:
                image = self.saga_module.render_rgb(view_idx)
            
            # 如果RGB也失败，则不显示
            if image is None:
                self.log("无法渲染SAGA图像，请检查模型状态", "error")
                return
            
            # 显示图像
            self.saga_canvas.set_image(image)
            
            # 如果有SAGA点提示，已在ImageCanvas中显示（通过click_points）
            
        except Exception as e:
            self.log(f"更新SAGA可视化失败: {str(e)}", "error")
            traceback.print_exc()
            # 尝试显示RGB图像作为fallback
            try:
                if self.model_loaded and self.saga_module:
                    fallback_image = self.saga_module.render_rgb(self.current_view_idx)
                    if fallback_image is not None:
                        self.saga_canvas.set_image(fallback_image)
            except:
                pass
    
    # ========== 3DGS编辑相关方法 ==========
    
    def toggle_edit_mode(self, enabled):
        """切换编辑模式"""
        if not EDITOR_AVAILABLE:
            return
        
        if enabled:
            # 进入编辑模式：显示右侧编辑工具栏
            if hasattr(self, 'right_edit_panel'):
                self.right_edit_panel.show()
            self.log("已进入编辑模式")
        else:
            # 退出编辑模式：隐藏右侧编辑工具栏
            if hasattr(self, 'right_edit_panel'):
                self.right_edit_panel.hide()
            # 取消所有选择工具
            if hasattr(self, 'select_box_btn'):
                self.select_box_btn.setChecked(False)
            if hasattr(self, 'select_brush_btn'):
                self.select_brush_btn.setChecked(False)
            if hasattr(self, 'select_lasso_btn'):
                self.select_lasso_btn.setChecked(False)
            # 取消编辑类型
            if hasattr(self, 'edit_3d_viewer'):
                self.edit_3d_viewer.set_selection_mode(False, None)
            self.log("已退出编辑模式")
    
    def set_view_mode(self, mode):
        """设置视图显示模式
        
        Args:
            mode: 'splat', 'point', 'depth', 'ellipsoids'
        """
        self.current_view_mode = mode
        self.log(f"视图模式切换为: {mode}")
        # TODO: 实际更新3D视图的渲染模式
        # 这里需要根据实际的3D viewer实现来调用相应的方法
        if hasattr(self, 'edit_3d_viewer') and self.edit_3d_viewer:
            # 如果有更新视图模式的方法，在这里调用
            pass
    
    def set_edit_type(self, edit_type):
        """设置编辑类型"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        self.current_edit_type = edit_type
        self.gaussian_editor.set_edit_type(edit_type)
        
        # 更新按钮状态（使用tool_buttons字典）
        if hasattr(self, 'tool_buttons'):
            # 创建编辑类型到按钮名称的映射
            type_to_button = {
                EditType.NONE: "选择",
                EditType.BOX: "矩形",
                EditType.BRUSH: "笔刷",
                EditType.LASSO: "套索",
            }
            
            # 设置对应按钮为选中状态，由于按钮组是互斥的，其他按钮会自动取消选中
            button_name = type_to_button.get(edit_type)
            if button_name and button_name in self.tool_buttons:
                self.tool_buttons[button_name].setChecked(True)
        
        # 设置3D视图的选择模式
        if hasattr(self, 'edit_3d_viewer'):
            # 映射EditType到选择类型字符串
            type_mapping = {
                EditType.BOX: 'box',
                EditType.BRUSH: 'brush',
                EditType.LASSO: 'lasso',
                EditType.NONE: 'none'
            }
            type_name = type_mapping.get(edit_type, 'box')
            self.edit_3d_viewer.set_selection_mode(edit_type != EditType.NONE, type_name)
            
            # 立即更新相机参数
            self.update_viewer_camera_params()
            
            self.log(f"编辑模式已切换为: {type_name}", "info")
    
    def edit_select_all(self):
        """全选"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.select_all()
            self.update_edit_view()
            self.log("已全选所有点", "info")
        except Exception as e:
            self.log(f"全选失败: {str(e)}", "error")
            self.show_message("错误", f"全选失败: {str(e)}", "critical")
    
    def edit_select_inverse(self):
        """反选"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.select_inverse()
            self.update_edit_view()
            self.log("已反选", "info")
        except Exception as e:
            self.log(f"反选失败: {str(e)}", "error")
            self.show_message("错误", f"反选失败: {str(e)}", "critical")
    
    def edit_select_none(self):
        """取消选择"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.select_none()
            self.update_edit_view()
            self.log("已取消选择", "info")
        except Exception as e:
            self.log(f"取消选择失败: {str(e)}", "error")
            self.show_message("错误", f"取消选择失败: {str(e)}", "critical")
    
    def edit_delete_selected(self):
        """删除选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            selected_indices = self.gaussian_editor.get_selected_indices()
            if len(selected_indices) == 0:
                self.show_message("提示", "没有选中的点", "info")
                return
            
            # 确认对话框
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除 {len(selected_indices)} 个选中的点吗？\n此操作可以通过撤销恢复。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.gaussian_editor.delete_selected()
                self.update_edit_view()
                self.update_display_view()  # 更新2D视图
                self.log(f"已删除 {len(selected_indices)} 个点", "info")
        except Exception as e:
            self.log(f"删除失败: {str(e)}", "error")
            self.show_message("错误", f"删除失败: {str(e)}", "critical")
    
    def edit_undo(self):
        """撤销"""
        if not EDITOR_AVAILABLE:
            return
        
        try:
            undo_system = UndoRedoSystem()
            if undo_system.undo():
                self.update_edit_view()
                self.update_display_view()
                self.log("已撤销", "info")
            else:
                self.log("没有可撤销的操作", "warning")
        except Exception as e:
            self.log(f"撤销失败: {str(e)}", "error")
    
    def edit_redo(self):
        """重做"""
        if not EDITOR_AVAILABLE:
            return
        
        try:
            undo_system = UndoRedoSystem()
            if undo_system.redo():
                self.update_edit_view()
                self.update_display_view()
                self.log("已重做", "info")
            else:
                self.log("没有可重做的操作", "warning")
        except Exception as e:
            self.log(f"重做失败: {str(e)}", "error")
    
    def edit_reset(self):
        """重置所有选择和删除标记"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            reply = QMessageBox.question(
                self, "确认重置",
                "确定要重置所有选择和删除标记吗？\n此操作可以通过撤销恢复。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.gaussian_editor.reset_selection()
                self.update_edit_view()
                self.update_display_view()
                self.log("已重置", "info")
        except Exception as e:
            self.log(f"重置失败: {str(e)}", "error")
            self.show_message("错误", f"重置失败: {str(e)}", "critical")
    
    def update_viewer_camera_params(self):
        """更新3D视图的相机参数"""
        if not EDITOR_AVAILABLE or not hasattr(self, 'edit_3d_viewer') or self.edit_3d_viewer is None:
            return
        
        try:
            # 视图会自动从pyqtgraph GL视图获取相机参数
            # 这里确保视图在需要时更新参数
            self.edit_3d_viewer._update_camera_params()
        except Exception as e:
            self.log(f"更新相机参数失败: {e}", "warning")
    
    def update_edit_view(self):
        """更新编辑视图"""
        if hasattr(self, 'edit_3d_viewer') and self.edit_3d_viewer:
            self.update_viewer_camera_params()
            self.edit_3d_viewer.update_view()
    
    def update_display_view(self):
        """更新显示视图（重新渲染当前视图）"""
        if self.model_loaded:
            self.display_current_view()
    
    def create_icon_toolbar(self, layout):
        """创建图标化工具栏（DIVSHOT风格）"""
        # 定义图标按钮样式
        icon_btn_style = """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px;
                color: #B0B0B0;
                font-size: 24px;
                min-width: 48px;
                min-height: 48px;
                max-width: 48px;
                max-height: 48px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                color: #FFFFFF;
            }
            QPushButton:checked {
                background-color: #0E639C;
                color: #FFFFFF;
            }
        """
        
        # 选择工具组（使用实例变量以防止被垃圾回收）
        self.select_group = QButtonGroup()
        self.select_group.setExclusive(True)
        
        # 图标工具（使用Unicode字符作为图标）
        tools = [
            ("🖱", "选择", lambda: self.set_edit_type(EditType.NONE if EDITOR_AVAILABLE else None), "选择模式"),
            ("⬜", "矩形", lambda: self.set_edit_type(EditType.BOX), "矩形框选"),
            ("🖌", "笔刷", lambda: self.set_edit_type(EditType.BRUSH), "笔刷工具"),
            ("⭕", "套索", lambda: self.set_edit_type(EditType.LASSO), "套索选择"),
        ]
        
        self.tool_buttons = {}
        for icon, name, callback, tooltip in tools:
            btn = QPushButton(icon)
            btn.setStyleSheet(icon_btn_style)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)
            self.select_group.addButton(btn)
            layout.addWidget(btn)
            self.tool_buttons[name] = btn
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #404040; margin: 4px 8px;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        
        # 操作工具（非互斥）
        operation_style = icon_btn_style.replace("QPushButton:checked", "QPushButton:pressed")
        
        operations = [
            ("👁", "显示/隐藏", self.toggle_visibility, "切换可见性"),
            ("🔒", "锁定/解锁", self.toggle_lock, "切换锁定"),
            ("🔓", "解锁全部", self.edit_unlock_all, "解锁所有锁定的点"),
            ("🎨", "颜色", self.open_color_panel, "颜色编辑"),
        ]
        
        # 如果多模式渲染器可用，添加渲染模式切换按钮
        if MULTI_MODE_AVAILABLE:
            operations.append(("🔮", "渲染模式", self.open_render_mode_menu, "切换渲染模式"))
        
        
        for icon, name, callback, tooltip in operations:
            btn = QPushButton(icon)
            btn.setStyleSheet(operation_style)
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)
            layout.addWidget(btn)
            self.tool_buttons[name] = btn
        
        layout.addStretch()
        
        # 底部工具
        bottom_tools = [
            ("ℹ", "信息", self.toggle_info_panel, "显示/隐藏信息面板"),
        ]
        
        for icon, name, callback, tooltip in bottom_tools:
            btn = QPushButton(icon)
            btn.setStyleSheet(operation_style)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            layout.addWidget(btn)
            self.tool_buttons[name] = btn
    
    def toggle_visibility(self):
        """切换可见性（快速操作）"""
        if not EDITOR_AVAILABLE:
            self.log("编辑器功能未启用", "error")
            self.show_message("错误", "编辑器功能未启用", "critical")
            return
            
        if not self.gaussian_editor:
            self.log("编辑器未初始化，请先加载模型", "warning")
            self.show_message("提示", "请先加载模型", "warning")
            return
            
        selected = self.gaussian_editor.get_selected_indices()
        if len(selected) == 0:
            self.log("请先选择要切换可见性的点", "warning")
            self.show_message("提示", "请先选择要切换可见性的点", "info")
            return
        
        # 检查选中点的状态，实现真正的切换
        if not hasattr(self.gaussian_editor.gaussians, '_state_flags'):
            # 如果没有状态标志，直接隐藏
            self.log(f"隐藏 {len(selected)} 个点", "info")
            self.edit_hide_selected()
            return
        
        flags = self.gaussian_editor.gaussians._state_flags
        # 检查选中的点中是否有隐藏的点
        selected_flags = flags[selected]
        has_hidden = torch.any((selected_flags & HIDE_STATE) != 0).item()
        
        if has_hidden:
            # 如果有隐藏的点，则显示它们
            self.log(f"显示 {len(selected)} 个点", "info")
            self.edit_unhide_selected()
        else:
            # 如果没有隐藏的点，则隐藏它们
            self.log(f"隐藏 {len(selected)} 个点", "info")
            self.edit_hide_selected()
    
    def toggle_lock(self):
        """切换锁定（快速操作）"""
        if not EDITOR_AVAILABLE:
            self.log("编辑器功能未启用", "error")
            self.show_message("错误", "编辑器功能未启用", "critical")
            return
            
        if not self.gaussian_editor:
            self.log("编辑器未初始化，请先加载模型", "warning")
            self.show_message("提示", "请先加载模型", "warning")
            return
            
        selected = self.gaussian_editor.get_selected_indices()
        if len(selected) == 0:
            self.log("请先选择要切换锁定状态的点", "warning")
            self.show_message("提示", "请先选择要切换锁定状态的点", "info")
            return
        
        # 检查选中点的状态，实现真正的切换
        if not hasattr(self.gaussian_editor.gaussians, '_state_flags'):
            # 如果没有状态标志，直接锁定
            self.log(f"锁定 {len(selected)} 个点", "info")
            self.edit_lock_selected()
            return
        
        flags = self.gaussian_editor.gaussians._state_flags
        # 检查选中的点中是否有锁定的点
        selected_flags = flags[selected]
        has_locked = torch.any((selected_flags & LOCK_STATE) != 0).item()
        
        if has_locked:
            # 如果有锁定的点，则解锁它们
            self.log(f"解锁 {len(selected)} 个点", "info")
            self.edit_unlock_selected()
        else:
            # 如果没有锁定的点，则锁定它们
            self.log(f"锁定 {len(selected)} 个点", "info")
            self.edit_lock_selected()
    
    def open_color_panel(self):
        """打开编辑面板（已删除独立颜色标签）"""
        if hasattr(self, 'right_edit_panel'):
            # 切换到统一的编辑工具标签
            if hasattr(self, 'edit_panel_tabs'):
                self.edit_panel_tabs.setCurrentIndex(0)  # 编辑工具标签
            self.right_edit_panel.show()
    
   
    
    def toggle_info_panel(self):
        """打开编辑工具"""
        if hasattr(self, 'right_edit_panel'):
            if self.right_edit_panel.isVisible():
                self.right_edit_panel.hide()
            else:
                self.right_edit_panel.show()
    
    def open_render_mode_menu(self):
        """打开渲染模式菜单（多模式渲染器）"""
        if not MULTI_MODE_AVAILABLE:
            return
        
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import QPoint
        
        # 创建菜单
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2B2B2B;
                border: 1px solid #404040;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                color: #B0B0B0;
            }
            QMenu::item:selected {
                background-color: #0E639C;
                color: #FFFFFF;
            }
        """)
        
        # 添加渲染模式选项
        for mode in RenderMode:
            # 如果是MESH模式且没有生成mesh，跳过该选项
            if mode == RenderMode.MESH:
                if not hasattr(self, 'edit_3d_viewer') or self.edit_3d_viewer is None:
                    continue
                if not hasattr(self.edit_3d_viewer, 'cached_mesh') or self.edit_3d_viewer.cached_mesh is None:
                    continue
            
            action = menu.addAction(f"{mode.name} - {self._get_mode_description(mode)}")
            action.triggered.connect(lambda checked, m=mode: self._change_render_mode(m))
        
        # 在按钮下方显示菜单
        if "渲染模式" in self.tool_buttons:
            btn = self.tool_buttons["渲染模式"]
            pos = btn.mapToGlobal(QPoint(0, btn.height()))
            menu.exec_(pos)
    
    def _get_mode_description(self, mode):
        """获取渲染模式描述"""
        descriptions = {
            RenderMode.SPLAT: "标准高斯Splat（基于尺度的半透明渲染）",
            RenderMode.POINT_CLOUD: "点云显示",
            RenderMode.DEPTH: "深度可视化",
            RenderMode.NORMAL: "法线可视化",
            RenderMode.ELLIPSOIDS: "椭球体显示",
            RenderMode.CENTERS: "中心点显示",
            RenderMode.MESH: "Mesh网格显示",
        }
        return descriptions.get(mode, "")
    
    def _change_render_mode(self, mode):
        """切换渲染模式"""
        if hasattr(self, 'edit_3d_viewer') and hasattr(self.edit_3d_viewer, 'set_render_mode'):
            self.edit_3d_viewer.set_render_mode(mode)
            self.log(f"渲染模式已切换: {mode.name}")
    
    def update_3d_viewer_with_segmentation(self):
        """更新3D渲染器显示分割后的模型（线程安全版本）"""
        if not hasattr(self, 'final_mask') or self.final_mask is None:
            self.log("没有可用的分割结果", "warning")
            return False
        
        if not self.model_loaded or self.gaussians is None:
            self.log("没有加载的模型", "warning")
            return False
        
        try:
            self.log("正在准备3D渲染器对比功能...")
            
            # 保存原始模型引用（如果还没有保存）
            if self.original_gaussians is None:
                self.original_gaussians = self.gaussians
                self.log("已保存原始模型引用用于对比")
            
            # 注意：我们不在这里创建分割后的模型，而是在用户点击按钮时动态创建
            # 这样可以避免内存和线程安全问题
            
            # 启用对比按钮
            if hasattr(self, 'comparison_btn'):
                self.comparison_btn.setEnabled(True)
                self.comparison_btn.setChecked(False)
                self.comparison_btn.setText("显示分割后模型")
            
            # 自动切换到3D渲染标签页
            if hasattr(self, 'display_tabs'):
                self.display_tabs.setCurrentIndex(0)  # 3D渲染是第一个标签页
            
            self.log("3D渲染器准备完成，可以使用'显示分割后模型'按钮进行对比", "info")
            return True
            
        except Exception as e:
            self.log(f"更新3D渲染器失败: {str(e)}", "error")
            traceback.print_exc()
            return False
    
    def toggle_segmentation_comparison(self):
        """切换显示原始模型和分割后的模型（延迟加载版本）"""
        if not hasattr(self, 'original_gaussians') or self.original_gaussians is None:
            self.log("没有原始模型", "warning")
            return
        
        if not hasattr(self, 'final_mask') or self.final_mask is None:
            self.log("没有分割掩码", "warning")
            return
        
        try:
            # 切换显示状态
            self.is_showing_segmented = not self.is_showing_segmented
            
            if self.is_showing_segmented:
                # 第一次切换到分割后模型时，才创建它
                if not hasattr(self, 'segmented_gaussians') or self.segmented_gaussians is None:
                    self.log("正在创建分割后的模型...")
                    
                    # 创建临时文件来保存分割后的模型
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    segmented_ply_path = os.path.join(temp_dir, "segmented.ply")
                    
                    try:
                        # 检查并处理不同类型的掩码
                        total_gaussians = self.original_gaussians.get_xyz.shape[0]
                        final_mask_np = self.final_mask.cpu().numpy() if isinstance(self.final_mask, torch.Tensor) else np.array(self.final_mask)
                        
                        # 判断是布尔掩码还是索引数组
                        is_bool_mask = final_mask_np.dtype == bool
                        
                        if is_bool_mask:
                            # 布尔掩码处理
                            self.log(f"检测到布尔掩码类型（SAGA分割）", "info")
                            
                            if len(final_mask_np) != total_gaussians:
                                self.log(f"警告: 布尔掩码大小不匹配", "warning")
                                self.log(f"  掩码大小: {len(final_mask_np)}, 模型大小: {total_gaussians}", "warning")
                                
                                # 尝试调整掩码大小
                                if len(final_mask_np) > total_gaussians:
                                    # 掩码太大，截断
                                    self.log(f"  正在截断掩码到模型大小...", "info")
                                    final_mask_np = final_mask_np[:total_gaussians]
                                else:
                                    # 掩码太小，扩展（新增部分设为False）
                                    self.log(f"  正在扩展掩码到模型大小...", "info")
                                    extended_mask = np.zeros(total_gaussians, dtype=bool)
                                    extended_mask[:len(final_mask_np)] = final_mask_np
                                    final_mask_np = extended_mask
                                
                                self.log(f"  调整后的掩码大小: {len(final_mask_np)}", "info")
                            
                            # 检查是否有选中的点
                            num_selected = np.sum(final_mask_np)
                            if num_selected == 0:
                                raise ValueError("布尔掩码中没有选中任何点")
                            
                            self.log(f"  选中的高斯点数: {num_selected}", "info")
                            filtered_mask = torch.from_numpy(final_mask_np).to(self.final_mask.device if isinstance(self.final_mask, torch.Tensor) else 'cpu')
                        else:
                            # 索引数组处理（SAGS分割）
                            self.log(f"检测到索引数组类型（SAGS分割）", "info")
                            
                            # 过滤有效索引
                            valid_mask = final_mask_np[final_mask_np < total_gaussians]
                            invalid_count = len(final_mask_np) - len(valid_mask)
                            
                            if invalid_count > 0:
                                self.log(f"警告: 分割掩码中有 {invalid_count} 个索引超出范围，已自动过滤", "warning")
                                self.log(f"  原始掩码大小: {len(final_mask_np)}, 有效掩码大小: {len(valid_mask)}", "info")
                            
                            if len(valid_mask) == 0:
                                raise ValueError("没有有效的分割索引")
                            
                            filtered_mask = torch.from_numpy(valid_mask).to(self.final_mask.device if isinstance(self.final_mask, torch.Tensor) else 'cpu')
                        
                        # 保存分割后的高斯点到PLY文件
                        save_gs(self.original_gaussians, filtered_mask, segmented_ply_path)
                        
                        # 加载分割后的模型
                        self.segmented_gaussians = GaussianModel(3)  # sh_degree=3
                        self.segmented_gaussians.load_ply(segmented_ply_path)
                        
                        self.log("分割后的模型创建成功")
                    except Exception as e:
                        self.log(f"创建分割后模型失败: {str(e)}", "error")
                        self.is_showing_segmented = False  # 回滚状态
                        raise
                    finally:
                        # 清理临时文件
                        try:
                            if os.path.exists(segmented_ply_path):
                                os.remove(segmented_ply_path)
                            if os.path.exists(temp_dir):
                                os.rmdir(temp_dir)
                        except:
                            pass
                
                # 切换到分割后的模型
                self.gaussians = self.segmented_gaussians
                if hasattr(self, 'comparison_btn'):
                    self.comparison_btn.setChecked(True)
                    self.comparison_btn.setText("显示原始模型")
                self.log("已切换到分割后的模型", "info")
            else:
                # 切换回原始模型
                self.gaussians = self.original_gaussians
                if hasattr(self, 'comparison_btn'):
                    self.comparison_btn.setChecked(False)
                    self.comparison_btn.setText("显示分割后模型")
                self.log("已切换到原始模型", "info")
            
            # 更新3D查看器（仅在编辑器模式可用时）
            if EDITOR_AVAILABLE and hasattr(self, 'edit_3d_viewer') and self.edit_3d_viewer is not None:
                if hasattr(self, 'gaussian_editor') and self.gaussian_editor is not None:
                    # 为当前模型创建编辑器
                    if self.is_showing_segmented:
                        # 为分割后的模型创建新的编辑器
                        if not hasattr(self, 'segmented_editor') or self.segmented_editor is None:
                            self.segmented_editor = GaussianEditor(self.segmented_gaussians)
                        self.edit_3d_viewer.set_gaussians(self.segmented_gaussians, self.segmented_editor)
                    else:
                        # 使用原始编辑器
                        self.edit_3d_viewer.set_gaussians(self.original_gaussians, self.gaussian_editor)
                    self.log("3D查看器已更新")
            
            # 更新当前视图显示
            if hasattr(self, 'current_view_idx') and hasattr(self, 'display_current_view'):
                self.display_current_view()
            
        except Exception as e:
            self.log(f"切换模型显示失败: {str(e)}", "error")
            traceback.print_exc()
            # 回滚到原始模型
            if hasattr(self, 'original_gaussians'):
                self.gaussians = self.original_gaussians
                self.is_showing_segmented = False
                if hasattr(self, 'comparison_btn'):
                    self.comparison_btn.setChecked(False)
                    self.comparison_btn.setText("显示分割后模型")
    
    def create_transform_and_settings_tab(self):
        """创建变换和设置合并标签页"""
        combined_widget = QWidget()
        combined_layout = QVBoxLayout(combined_widget)
        combined_layout.setContentsMargins(12, 12, 12, 12)
        combined_layout.setSpacing(12)
        
        # 选择操作
        select_group_layout = QVBoxLayout()
        select_group_layout.setSpacing(6)
        
        select_label = QLabel("选择操作")
        select_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold;")
        select_group_layout.addWidget(select_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        
        all_btn = QPushButton("全选")
        all_btn.clicked.connect(self.edit_select_all)
        btn_layout.addWidget(all_btn)
        
        inv_btn = QPushButton("反选")
        inv_btn.clicked.connect(self.edit_select_inverse)
        btn_layout.addWidget(inv_btn)
        
        none_btn = QPushButton("清除")
        none_btn.clicked.connect(self.edit_select_none)
        btn_layout.addWidget(none_btn)
        
        select_group_layout.addLayout(btn_layout)
        combined_layout.addLayout(select_group_layout)
        
        # 变换控制
        trans_label = QLabel("变换 (X, Y, Z)")
        trans_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold; margin-top: 8px;")
        combined_layout.addWidget(trans_label)
        
        trans_input = QHBoxLayout()
        self.translate_x_input = QLineEdit()
        self.translate_x_input.setPlaceholderText("X")
        self.translate_y_input = QLineEdit()
        self.translate_y_input.setPlaceholderText("Y")
        self.translate_z_input = QLineEdit()
        self.translate_z_input.setPlaceholderText("Z")
        trans_input.addWidget(self.translate_x_input)
        trans_input.addWidget(self.translate_y_input)
        trans_input.addWidget(self.translate_z_input)
        combined_layout.addLayout(trans_input)
        
        trans_btn = QPushButton("应用平移")
        trans_btn.clicked.connect(self.apply_precise_translate)
        combined_layout.addWidget(trans_btn)
        
        # 缩放控制
        scale_label = QLabel("缩放 (X, Y, Z)")
        scale_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold; margin-top: 8px;")
        combined_layout.addWidget(scale_label)
        
        scale_input = QHBoxLayout()
        self.scale_x_input = QLineEdit()
        self.scale_x_input.setPlaceholderText("X")
        self.scale_y_input = QLineEdit()
        self.scale_y_input.setPlaceholderText("Y")
        self.scale_z_input = QLineEdit()
        self.scale_z_input.setPlaceholderText("Z")
        scale_input.addWidget(self.scale_x_input)
        scale_input.addWidget(self.scale_y_input)
        scale_input.addWidget(self.scale_z_input)
        combined_layout.addLayout(scale_input)
        
        scale_btn = QPushButton("应用缩放")
        scale_btn.clicked.connect(self.apply_precise_scale)
        combined_layout.addWidget(scale_btn)
        
        # 旋转控制
        rotate_label = QLabel("旋转 (X°, Y°, Z°)")
        rotate_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold; margin-top: 8px;")
        combined_layout.addWidget(rotate_label)
        
        rotate_input = QHBoxLayout()
        self.rotate_x_input = QLineEdit()
        self.rotate_x_input.setPlaceholderText("X°")
        self.rotate_y_input = QLineEdit()
        self.rotate_y_input.setPlaceholderText("Y°")
        self.rotate_z_input = QLineEdit()
        self.rotate_z_input.setPlaceholderText("Z°")
        rotate_input.addWidget(self.rotate_x_input)
        rotate_input.addWidget(self.rotate_y_input)
        rotate_input.addWidget(self.rotate_z_input)
        combined_layout.addLayout(rotate_input)
        
        rotate_btn = QPushButton("应用旋转")
        rotate_btn.clicked.connect(self.apply_precise_rotate)
        combined_layout.addWidget(rotate_btn)
        
        # 其他操作
        other_label = QLabel("其他操作")
        other_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold; margin-top: 8px;")
        combined_layout.addWidget(other_label)
        
        other_btns = QHBoxLayout()
        dup_btn = QPushButton("复制")
        dup_btn.clicked.connect(self.edit_duplicate)
        other_btns.addWidget(dup_btn)
        
        exp_btn = QPushButton("导出")
        exp_btn.clicked.connect(self.edit_export_selected)
        other_btns.addWidget(exp_btn)
        
        del_btn = QPushButton("删除")
        del_btn.setStyleSheet("background-color: #C75050; color: white;")
        del_btn.clicked.connect(self.edit_delete_selected)
        other_btns.addWidget(del_btn)
        
        combined_layout.addLayout(other_btns)
        
        # 添加分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #404040; margin: 12px 0px;")
        separator.setFixedHeight(1)
        combined_layout.addWidget(separator)
        
        # 统计信息（从设置标签页合并）
        stats_label = QLabel("统计信息")
        stats_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold;")
        combined_layout.addWidget(stats_label)
        
        self.stats_label = QLabel("未加载模型")
        self.stats_label.setStyleSheet("color: #B0B0B0; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        combined_layout.addWidget(self.stats_label)
        
        refresh_btn = QPushButton("刷新统计")
        refresh_btn.clicked.connect(self.refresh_selection_stats)
        combined_layout.addWidget(refresh_btn)
        
        # 历史记录（从设置标签页合并）
        history_label = QLabel("历史记录")
        history_label.setStyleSheet("color: #E0E0E0; font-size: 12px; font-weight: bold; margin-top: 12px;")
        combined_layout.addWidget(history_label)
        
        history_btns = QHBoxLayout()
        undo_btn = QPushButton("撤销")
        undo_btn.setShortcut("Ctrl+Z")
        undo_btn.clicked.connect(self.edit_undo)
        history_btns.addWidget(undo_btn)
        
        redo_btn = QPushButton("重做")
        redo_btn.setShortcut("Ctrl+Y")
        redo_btn.clicked.connect(self.edit_redo)
        history_btns.addWidget(redo_btn)
        
        reset_btn = QPushButton("重置")
        reset_btn.clicked.connect(self.edit_reset)
        history_btns.addWidget(reset_btn)
        
        combined_layout.addLayout(history_btns)
        
        combined_layout.addStretch()
        
        self.edit_panel_tabs.addTab(combined_widget, "编辑工具")
    
    def on_edit_selection_changed(self):
        """编辑选择改变时的回调"""
        self.log("选择已更新", "info")
        # 更新统计信息
        if hasattr(self, 'refresh_selection_stats'):
            self.refresh_selection_stats()
        # 确保锁定点高亮也被更新
        if hasattr(self, 'edit_3d_viewer') and self.edit_3d_viewer:
            # 强制更新选择高亮（包括锁定点）
            if hasattr(self.edit_3d_viewer, '_update_selection_highlight'):
                self.edit_3d_viewer._update_selection_highlight()
    
    # ========== 编辑相关方法结束 ==========

    # ========== 编辑高级操作绑定 ==========
    def edit_translate(self, delta):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            self.gaussian_editor.translate_selected(delta)
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"平移失败: {e}", "error")

    def edit_scale(self, scale):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            self.gaussian_editor.scale_selected(scale)
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"缩放失败: {e}", "error")

    def edit_rotate(self, quat):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            self.gaussian_editor.rotate_selected(quat)
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"旋转失败: {e}", "error")

    def edit_adjust_color(self, delta_rgb, brightness):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            self.gaussian_editor.adjust_color_selected(np.array(delta_rgb, dtype=np.float32), float(brightness))
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"颜色调整失败: {e}", "error")

    def edit_duplicate(self):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            self.gaussian_editor.duplicate_selected()
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"复制失败: {e}", "error")

    def edit_export_selected(self):
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, "导出选中到PLY", "selected.ply", "PLY Files (*.ply)")
            if save_path:
                self.gaussian_editor.export_selected(save_path)
                self.show_message("成功", f"已导出: {save_path}", "info")
        except Exception as e:
            self.log(f"导出失败: {e}", "error")
    
    def edit_separate_selected(self):
        """分离选中点（复制到新实例并删除原选中）"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            selected_indices = self.gaussian_editor.get_selected_indices()
            if len(selected_indices) == 0:
                self.show_message("提示", "没有选中的点", "info")
                return
            
            # 确保索引在正确的设备上
            device = self.gaussians.get_xyz.device
            if selected_indices.device != device:
                selected_indices = selected_indices.to(device)
            
            # 先复制（这会创建新的点）
            self.gaussian_editor.duplicate_selected()
            
            # 等待复制完成（duplicate_selected是同步的）
            # 然后删除原选中（复制后新点会追加到末尾，原选中仍在原位置）
            self.gaussian_editor.delete_selected()
            
            self.update_edit_view()
            self.update_display_view()
            self.log(f"已分离 {len(selected_indices)} 个点", "info")
        except Exception as e:
            self.log(f"分离失败: {e}", "error")
            self.show_message("错误", f"分离失败: {e}", "critical")
    
    def on_color_adjust(self):
        """颜色滑块改变时的回调"""
        r_val = self.color_r_slider.value() / 100.0
        g_val = self.color_g_slider.value() / 100.0
        b_val = self.color_b_slider.value() / 100.0
        brightness_val = self.brightness_slider.value() / 100.0
        
        self.color_r_label.setText(f"{r_val:.2f}")
        self.color_g_label.setText(f"{g_val:.2f}")
        self.color_b_label.setText(f"{b_val:.2f}")
        self.brightness_label.setText(f"{brightness_val:.2f}")
    
    def apply_color_adjustment(self):
        """应用颜色调整"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        try:
            # 检查是否是绘制模式
            if hasattr(self, 'paint_mode_check') and self.paint_mode_check.isChecked():
                # 颜色绘制模式
                if hasattr(self, 'paint_color'):
                    mix_weight = self.mix_weight_slider.value() / 100.0 if hasattr(self, 'mix_weight_slider') else 0.5
                    self.gaussian_editor.paint_color(self.paint_color, mix_weight)
                    self.log("颜色绘制已应用", "info")
                else:
                    self.show_message("提示", "请先选择颜色", "warning")
                    return
            else:
                # 颜色调整模式
                r_val = self.color_r_slider.value() / 100.0
                g_val = self.color_g_slider.value() / 100.0
                b_val = self.color_b_slider.value() / 100.0
                brightness_val = self.brightness_slider.value() / 100.0
                transparency_val = self.transparency_slider.value() / 100.0 if hasattr(self, 'transparency_slider') else 0.0
                
                self.gaussian_editor.adjust_color(
                    [r_val, g_val, b_val],
                    brightness_val,
                    transparency_val
                )
                self.log("颜色调整已应用", "info")
            
            self.update_edit_view()
            self.update_display_view()
        except Exception as e:
            self.log(f"颜色操作失败: {e}", "error")
    
    def reset_color_sliders(self):
        """重置颜色滑块"""
        self.color_r_slider.setValue(0)
        self.color_g_slider.setValue(0)
        self.color_b_slider.setValue(0)
        self.brightness_slider.setValue(0)
        if hasattr(self, 'transparency_slider'):
            self.transparency_slider.setValue(0)
        self.on_color_adjust()
    
    def edit_hide_selected(self):
        """隐藏选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.hide_selected()
            self.update_edit_view()
            self.update_display_view()  # 更新主渲染视图
            self.refresh_selection_stats()
            self.log("已隐藏选中的点", "info")
        except Exception as e:
            self.log(f"隐藏失败: {str(e)}", "error")
            self.show_message("错误", f"隐藏失败: {str(e)}", "critical")
    
    def edit_unhide_selected(self):
        """显示选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.unhide_selected()
            self.update_edit_view()
            self.update_display_view()  # 更新主渲染视图
            self.refresh_selection_stats()
            self.log("已显示选中的点", "info")
        except Exception as e:
            self.log(f"显示失败: {str(e)}", "error")
            self.show_message("错误", f"显示失败: {str(e)}", "critical")
    
    def edit_lock_selected(self):
        """锁定选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.lock_selected()
            self.refresh_selection_stats()
            self.log("已锁定选中的点", "info")
        except Exception as e:
            self.log(f"锁定失败: {str(e)}", "error")
            self.show_message("错误", f"锁定失败: {str(e)}", "critical")
    
    def edit_unlock_selected(self):
        """解锁选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            self.gaussian_editor.unlock_selected()
            self.refresh_selection_stats()
            self.log("已解锁选中的点", "info")
        except Exception as e:
            self.log(f"解锁失败: {str(e)}", "error")
            self.show_message("错误", f"解锁失败: {str(e)}", "critical")
    
    def edit_unlock_all(self):
        """解锁所有锁定的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            # 获取当前锁定点数量
            if hasattr(self.gaussian_editor.gaussians, '_state_flags'):
                flags = self.gaussian_editor.gaussians._state_flags
                locked_count = torch.sum((flags & LOCK_STATE) != 0).item()
                
                if locked_count == 0:
                    self.log("没有锁定的点需要解锁", "info")
                    self.show_message("提示", "没有锁定的点", "info")
                    return
                
                # 执行解锁操作
                self.gaussian_editor.unlock_all()
                self.update_display_view()  # 更新主渲染视图
                self.refresh_selection_stats()
                self.log(f"已解锁所有 {locked_count} 个锁定的点", "info")
                self.show_message("成功", f"已解锁 {locked_count} 个点", "info")
            else:
                self.log("没有锁定的点", "info")
        except Exception as e:
            self.log(f"解锁所有点失败: {str(e)}", "error")
            self.show_message("错误", f"解锁所有点失败: {str(e)}", "critical")
    
    def apply_filters(self):
        """应用过滤器"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            self.show_message("错误", "编辑器未初始化", "critical")
            return
        
        try:
            # 解析半径过滤器
            min_radius = None
            max_radius = None
            if hasattr(self, 'min_radius_input') and self.min_radius_input.text():
                min_radius = float(self.min_radius_input.text())
            if hasattr(self, 'max_radius_input') and self.max_radius_input.text():
                max_radius = float(self.max_radius_input.text())
            
            # 解析不透明度过滤器
            min_opacity = None
            max_opacity = None
            if hasattr(self, 'min_opacity_input') and self.min_opacity_input.text():
                min_opacity = float(self.min_opacity_input.text())
            if hasattr(self, 'max_opacity_input') and self.max_opacity_input.text():
                max_opacity = float(self.max_opacity_input.text())
            
            # 应用过滤器
            if min_radius is not None or max_radius is not None:
                self.gaussian_editor.filter_by_radius(min_radius, max_radius)
                self.log(f"已应用半径过滤: {min_radius} - {max_radius}", "info")
            
            if min_opacity is not None or max_opacity is not None:
                from gaussian_editor import EditSelectOpType
                self.gaussian_editor.filter_by_opacity(min_opacity, max_opacity, EditSelectOpType.ADD)
                self.log(f"已应用不透明度过滤: {min_opacity} - {max_opacity}", "info")
            
            self.update_edit_view()
            self.refresh_selection_stats()
        except ValueError as e:
            self.show_message("错误", "过滤器参数无效", "warning")
        except Exception as e:
            self.log(f"应用过滤器失败: {str(e)}", "error")
            self.show_message("错误", f"应用过滤器失败: {str(e)}", "critical")
    
    def toggle_paint_mode(self, enabled):
        """切换颜色绘制模式"""
        if enabled:
            # 启用绘制模式
            if hasattr(self, 'paint_color_btn'):
                self.paint_color_btn.setEnabled(True)
            if hasattr(self, 'mix_weight_slider'):
                self.mix_weight_slider.setEnabled(True)
            # 禁用RGB调整滑块
            self.color_r_slider.setEnabled(False)
            self.color_g_slider.setEnabled(False)
            self.color_b_slider.setEnabled(False)
            self.brightness_slider.setEnabled(False)
            if hasattr(self, 'transparency_slider'):
                self.transparency_slider.setEnabled(False)
            self.log("已启用颜色绘制模式", "info")
        else:
            # 禁用绘制模式
            if hasattr(self, 'paint_color_btn'):
                self.paint_color_btn.setEnabled(False)
            if hasattr(self, 'mix_weight_slider'):
                self.mix_weight_slider.setEnabled(False)
            # 启用RGB调整滑块
            self.color_r_slider.setEnabled(True)
            self.color_g_slider.setEnabled(True)
            self.color_b_slider.setEnabled(True)
            self.brightness_slider.setEnabled(True)
            if hasattr(self, 'transparency_slider'):
                self.transparency_slider.setEnabled(True)
            self.log("已禁用颜色绘制模式", "info")
    
    def select_paint_color(self):
        """选择绘制颜色"""
        color = QColorDialog.getColor()
        if color.isValid():
            self.paint_color = [color.redF(), color.greenF(), color.blueF()]
            if hasattr(self, 'paint_color_btn'):
                self.paint_color_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color.name()};
                        color: {'white' if color.lightness() < 128 else 'black'};
                        border: 1px solid #505050;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-size: 16px;
                    }}
                """)
            self.log(f"已选择颜色: RGB({self.paint_color[0]:.2f}, {self.paint_color[1]:.2f}, {self.paint_color[2]:.2f})", "info")
    
    def refresh_selection_stats(self):
        """刷新选择统计信息"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            if hasattr(self, 'stats_label'):
                self.stats_label.setText("未加载模型")
            return
        
        try:
            stats = self.gaussian_editor.get_selection_stats()
            stats_text = f"""总点数: {stats['total']}
已选中: {stats['selected']}
可见: {stats['visible']}
已隐藏: {stats['hidden']}
已锁定: {stats['locked']}
已删除: {stats['deleted']}"""
            if hasattr(self, 'stats_label'):
                self.stats_label.setText(stats_text)
        except Exception as e:
            self.log(f"刷新统计信息失败: {str(e)}", "error")
            if hasattr(self, 'stats_label'):
                self.stats_label.setText("统计信息不可用")
    
    def on_brush_thickness_changed(self, value):
        """笔刷粗细改变的回调"""
        if hasattr(self, 'brush_thickness_label'):
            self.brush_thickness_label.setText(str(value))
        if EDITOR_AVAILABLE and self.gaussian_editor:
            self.gaussian_editor.brush_thickness = value
            self.log(f"笔刷粗细已设置为: {value}", "info")
    
    def set_pivot_to_center(self):
        """将pivot点设置到选中点的中心"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            center = self.gaussian_editor.set_pivot_to_selection_center()
            if center is not None:
                self.log(f"Pivot已设置到选中点中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})", "info")
            else:
                self.show_message("提示", "请先选中一些点", "warning")
        except Exception as e:
            self.log(f"设置Pivot失败: {str(e)}", "error")
    
    def set_pivot_to_origin(self):
        """将pivot点设置到世界原点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            self.gaussian_editor.set_pivot_to_world_origin()
            self.log("Pivot已设置到世界原点 (0, 0, 0)", "info")
        except Exception as e:
            self.log(f"设置Pivot失败: {str(e)}", "error")
    
    def align_to_axis(self, axis):
        """对齐到坐标轴"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            self.gaussian_editor.align_selection_to_axis(axis)
            self.update_edit_view()
            self.log(f"已对齐到{axis.upper()}轴", "info")
        except Exception as e:
            self.log(f"对齐失败: {str(e)}", "error")
            self.show_message("错误", f"对齐失败: {str(e)}", "critical")
    
    def align_to_grid(self):
        """对齐到网格"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            grid_size = 0.1
            if hasattr(self, 'grid_size_input') and self.grid_size_input.text():
                grid_size = float(self.grid_size_input.text())
            
            self.gaussian_editor.align_selection_to_grid(grid_size)
            self.update_edit_view()
            self.log(f"已对齐到网格 (大小: {grid_size})", "info")
        except ValueError:
            self.show_message("错误", "网格大小必须是数字", "warning")
        except Exception as e:
            self.log(f"对齐失败: {str(e)}", "error")
            self.show_message("错误", f"对齐失败: {str(e)}", "critical")
    
    def distribute_evenly(self):
        """均匀分布选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            axis_map = {'X轴': 'x', 'Y轴': 'y', 'Z轴': 'z'}
            axis = axis_map.get(self.distribute_axis_combo.currentText(), 'x')
            
            self.gaussian_editor.distribute_selection_evenly(axis)
            self.update_edit_view()
            self.log(f"已沿{axis.upper()}轴均匀分布", "info")
        except Exception as e:
            self.log(f"均匀分布失败: {str(e)}", "error")
            self.show_message("错误", f"均匀分布失败: {str(e)}", "critical")
    
    def mirror_selection(self):
        """镜像选中的点"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            axis_map = {'X轴': 'x', 'Y轴': 'y', 'Z轴': 'z'}
            axis = axis_map.get(self.mirror_axis_combo.currentText(), 'x')
            
            self.gaussian_editor.mirror_selection(axis)
            self.update_edit_view()
            self.refresh_selection_stats()
            self.log(f"已沿{axis.upper()}轴镜像", "info")
        except Exception as e:
            self.log(f"镜像失败: {str(e)}", "error")
            self.show_message("错误", f"镜像失败: {str(e)}", "critical")
    
    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        将欧拉角转换为四元数
        
        Args:
            roll: 绕X轴旋转角度（弧度）
            pitch: 绕Y轴旋转角度（弧度）
            yaw: 绕Z轴旋转角度（弧度）
        
        Returns:
            四元数 [w, x, y, z]
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def apply_precise_translate(self):
        """应用精确平移"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            x = float(self.translate_x_input.text()) if self.translate_x_input.text() else 0.0
            y = float(self.translate_y_input.text()) if self.translate_y_input.text() else 0.0
            z = float(self.translate_z_input.text()) if self.translate_z_input.text() else 0.0
            
            translation = [x, y, z]
            self.gaussian_editor.transform_selected_with_undo(translation=translation)
            self.update_edit_view()
            self.log(f"已应用平移: ({x:.3f}, {y:.3f}, {z:.3f})", "info")
            
            # 清空输入框
            self.translate_x_input.clear()
            self.translate_y_input.clear()
            self.translate_z_input.clear()
        except ValueError:
            self.show_message("错误", "请输入有效的数值", "warning")
        except Exception as e:
            self.log(f"平移失败: {str(e)}", "error")
            self.show_message("错误", f"平移失败: {str(e)}", "critical")
    
    def apply_precise_scale(self):
        """应用精确缩放"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
            y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0
            z = float(self.scale_z_input.text()) if self.scale_z_input.text() else 1.0
            
            scale = [x, y, z]
            self.gaussian_editor.transform_selected_with_undo(scale=scale)
            self.update_edit_view()
            self.log(f"已应用缩放: ({x:.3f}, {y:.3f}, {z:.3f})", "info")
            
            # 清空输入框
            self.scale_x_input.clear()
            self.scale_y_input.clear()
            self.scale_z_input.clear()
        except ValueError:
            self.show_message("错误", "请输入有效的数值", "warning")
        except Exception as e:
            self.log(f"缩放失败: {str(e)}", "error")
            self.show_message("错误", f"缩放失败: {str(e)}", "critical")
    
    def apply_precise_rotate(self):
        """应用精确旋转"""
        if not EDITOR_AVAILABLE or not self.gaussian_editor:
            return
        
        try:
            # 获取欧拉角（度数）
            x_deg = float(self.rotate_x_input.text()) if self.rotate_x_input.text() else 0.0
            y_deg = float(self.rotate_y_input.text()) if self.rotate_y_input.text() else 0.0
            z_deg = float(self.rotate_z_input.text()) if self.rotate_z_input.text() else 0.0
            
            # 转换为弧度
            x_rad = np.radians(x_deg)
            y_rad = np.radians(y_deg)
            z_rad = np.radians(z_deg)
            
            # 转换为四元数
            quat = self.euler_to_quaternion(x_rad, y_rad, z_rad)
            
            self.gaussian_editor.transform_selected_with_undo(rotation=quat.tolist())
            self.update_edit_view()
            self.log(f"已应用旋转: ({x_deg:.1f}°, {y_deg:.1f}°, {z_deg:.1f}°)", "info")
            
            # 清空输入框
            self.rotate_x_input.clear()
            self.rotate_y_input.clear()
            self.rotate_z_input.clear()
        except ValueError:
            self.show_message("错误", "请输入有效的数值", "warning")
        except Exception as e:
            self.log(f"旋转失败: {str(e)}", "error")
            self.show_message("错误", f"旋转失败: {str(e)}", "critical")
    
    def apply_theme(self, theme_name):
        """应用主题
        
        Args:
            theme_name: 'dark' 或 'light'
        """
        self.current_theme = theme_name
        self.settings.setValue("theme", theme_name)
        
        if theme_name == 'dark':
            theme_style = ThemeManager.get_dark_theme()
            # 深色主题的颜色常量
            self.status_color = "#5BA3D8"  # 蓝色状态文字
            self.hint_color = "#B0B0B0"  # 灰色提示文字
            self.text_color = "#E0E0E0"  # 主要文字颜色
            self.bg_dark = "#2B2B2B"  # 深色背景
            self.bg_panel = "#252525"  # 面板背景
            self.border_color = "#404040"  # 边框颜色
            self.edit_title_bg = "#252525"
            self.edit_title_border = "#404040"
            self.group_bg = "#2B2B2B"
            self.info_bg = "#2B2B2B"
            self.info_border = "#404040"
            self.btn_primary = "#0E639C"
            self.btn_primary_hover = "#1177BB"
            self.btn_danger = "#C75050"
            self.btn_danger_hover = "#D96A6A"
        else:  # light
            theme_style = ThemeManager.get_light_theme()
            # 浅色主题的颜色常量
            self.status_color = "#2196F3"  # 蓝色状态文字
            self.hint_color = "#666666"  # 灰色提示文字
            self.text_color = "#212529"  # 主要文字颜色
            self.bg_dark = "#F0F0F0"  # 浅色背景
            self.bg_panel = "#FFFFFF"  # 面板背景
            self.border_color = "#CCCCCC"  # 边框颜色
            self.edit_title_bg = "#F0F0F0"
            self.edit_title_border = "#CCCCCC"
            self.group_bg = "#F9F9F9"
            self.info_bg = "#F5F5F5"
            self.info_border = "#CCCCCC"
            self.btn_primary = "#2196F3"
            self.btn_primary_hover = "#1976D2"
            self.btn_danger = "#DC3545"
            self.btn_danger_hover = "#C82333"
        
        # 应用全局样式
        if self.app:
            self.app.setStyleSheet(theme_style)
        
        # 更新自定义样式的组件
        self.update_component_styles()
        
        self.log(f"已切换到{'深色' if theme_name == 'dark' else '浅色'}主题", "info")
    
    def update_component_styles(self):
        """更新需要自定义样式的组件"""
        # 更新状态标签颜色
        if hasattr(self, 'video_info_label'):
            self.video_info_label.setStyleSheet(f"color: {self.status_color};")
        if hasattr(self, 'colmap_status_label'):
            self.colmap_status_label.setStyleSheet(f"color: {self.status_color};")
        if hasattr(self, 'training_status_label'):
            self.training_status_label.setStyleSheet(f"color: {self.status_color};")
        if hasattr(self, 'sog_training_status_label'):
            self.sog_training_status_label.setStyleSheet(f"color: {self.status_color};")
        if hasattr(self, 'saga_status_label'):
            self.saga_status_label.setStyleSheet(f"color: {self.status_color};")
        
        # 更新提示文字颜色
        hint_labels = []
        if hasattr(self, 'left_widget'):
            hint_labels = self.left_widget.findChildren(QLabel)
        
        for label in hint_labels:
            current_style = label.styleSheet()
            if current_style and ("gray" in current_style.lower() or "#B0B0B0" in current_style or "#666" in current_style):
                label.setStyleSheet(f"color: {self.hint_color};")
        
        # 更新编辑面板样式
        if hasattr(self, 'right_edit_panel'):
            self.right_edit_panel.setStyleSheet(f"""
                QWidget {{
                    background-color: {self.bg_dark};
                    border-left: 1px solid {self.border_color};
                }}
            """)
        
        # 更新编辑标题
        if hasattr(self, 'edit_title'):
            edit_title = self.findChild(QLabel, "edit_title")
            if edit_title:
                edit_title.setStyleSheet(f"""
                    QLabel {{
                        background-color: {self.edit_title_bg};
                        color: {self.text_color};
                        font-size: 13px;
                        font-weight: bold;
                        padding: 4px;
                    }}
                """)
        
        # 更新分组框样式（这些会通过全局样式自动更新，但我们可以确保它们正确）
        # 更新日志文本框
        if hasattr(self, 'log_text'):
            self.log_text.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {self.bg_panel};
                    color: {self.text_color};
                    border: 1px solid {self.border_color};
                }}
            """)
        
        # 更新颜色标签
        if hasattr(self, 'color_r_label'):
            self.color_r_label.setStyleSheet(f"color: {self.text_color}; font-size: 10px;")
            self.color_g_label.setStyleSheet(f"color: {self.text_color}; font-size: 10px;")
            self.color_b_label.setStyleSheet(f"color: {self.text_color}; font-size: 10px;")
            self.brightness_label.setStyleSheet(f"color: {self.text_color}; font-size: 10px;")
        
        # 更新顶部菜单栏
        if hasattr(self, 'top_menu_bar'):
            top_menu_bar = self.findChild(QWidget, "top_menu_bar")
            if top_menu_bar:
                top_menu_bar.setStyleSheet(f"""
                    QWidget {{
                        background-color: {self.bg_dark};
                        border-bottom: 1px solid {self.border_color};
                    }}
                """)
        
        # 更新分隔线颜色
        separators = self.findChildren(QFrame)
        for sep in separators:
            if sep.frameShape() == QFrame.VLine or sep.frameShape() == QFrame.HLine:
                sep.setStyleSheet(f"color: {self.border_color};")
        
        # 更新按钮样式（特殊按钮）
        if hasattr(self, 'view_mode_splat_btn'):
            base_button_style = f"""
                QPushButton {{
                    background-color: {'#424242' if self.current_theme == 'dark' else '#FFFFFF'};
                    color: {self.text_color};
                    border: 1px solid {self.border_color};
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 16px;
                }}
                QPushButton:checked {{
                    background-color: {self.btn_primary};
                    color: white;
                    border-color: {self.btn_primary};
                }}
                QPushButton:hover:!checked {{
                    background-color: {'#505050' if self.current_theme == 'dark' else '#F0F0F0'};
                }}
            """
            self.view_mode_splat_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'view_mode_point_btn'):
                self.view_mode_point_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'view_mode_depth_btn'):
                self.view_mode_depth_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'view_mode_ellipsoids_btn'):
                self.view_mode_ellipsoids_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'mouse_select_btn'):
                self.mouse_select_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'mouse_translate_btn'):
                self.mouse_translate_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'mouse_rotate_btn'):
                self.mouse_rotate_btn.setStyleSheet(base_button_style)
            if hasattr(self, 'mouse_scale_btn'):
                self.mouse_scale_btn.setStyleSheet(base_button_style)
        
        # 更新编辑模式按钮
        if hasattr(self, 'edit_mode_btn'):
            self.edit_mode_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {'#424242' if self.current_theme == 'dark' else '#FFFFFF'};
                    color: {self.text_color};
                    border: 1px solid {self.border_color};
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 16px;
                }}
                QPushButton:checked {{
                    background-color: #28a745;
                    color: white;
                    border-color: #28a745;
                }}
                QPushButton:hover:!checked {{
                    background-color: {'#505050' if self.current_theme == 'dark' else '#F0F0F0'};
                }}
            """)
        
        # 更新主要操作按钮
        primary_btn_style = f"""
            QPushButton {{
                background-color: {self.btn_primary};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {self.btn_primary_hover};
            }}
        """
        
        danger_btn_style = f"""
            QPushButton {{
                background-color: {self.btn_danger};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {self.btn_danger_hover};
            }}
        """
        
        # 更新删除按钮
        if hasattr(self, 'delete_btn'):
            self.delete_btn.setStyleSheet(danger_btn_style)
        
        # 更新重置按钮
        reset_btn = self.findChild(QPushButton)
        if reset_btn and "重置" in reset_btn.text():
            reset_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.btn_danger};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-weight: bold;
                    min-height: 35px;
                }}
                QPushButton:hover {{
                    background-color: {self.btn_danger_hover};
                }}
                QPushButton:pressed {{
                    background-color: {'#B53838' if self.current_theme == 'dark' else '#C82333'};
                }}
            """)
    
    # ==================== GIS相关方法 ====================
    
    def on_gis_preset_changed(self, preset_name: str):
        """预设位置改变时的回调"""
        if not GIS_AVAILABLE or preset_name == "自定义":
            return
        
        if preset_name in PRESET_LOCATIONS:
            location = PRESET_LOCATIONS[preset_name]
            self.gis_longitude_input.setValue(location['longitude'])
            self.gis_latitude_input.setValue(location['latitude'])
            self.gis_altitude_input.setValue(location.get('altitude', 0.0))
            self.log(f"已切换到预设位置: {preset_name}", "info")
    
    def save_gis_config(self):
        """保存GIS配置"""
        if not GIS_AVAILABLE:
            return
        
        config_path, _ = QFileDialog.getSaveFileName(
            self, "保存GIS配置", "", "JSON文件 (*.json)"
        )
        if not config_path:
            return
        
        # 创建或更新配置
        if self.geo_config is None:
            self.geo_config = GeoDataConfig(config_path)
        else:
            self.geo_config.config_path = config_path
        
        # 更新配置
        self.geo_config.update_origin(
            self.gis_longitude_input.value(),
            self.gis_latitude_input.value(),
            self.gis_altitude_input.value()
        )
        self.geo_config.update_scale(self.gis_scale_input.value())
        self.geo_config.update_rotation(self.gis_rotation_input.value())
        
        self.log(f"GIS配置已保存到: {config_path}", "info")
    
    def load_gis_config(self):
        """加载GIS配置"""
        if not GIS_AVAILABLE:
            return
        
        config_path, _ = QFileDialog.getOpenFileName(
            self, "加载GIS配置", "", "JSON文件 (*.json)"
        )
        if not config_path:
            return
        
        try:
            self.geo_config = GeoDataConfig(config_path)
            config = self.geo_config.config
            
            # 更新UI
            origin = config.get('origin', {})
            self.gis_longitude_input.setValue(origin.get('longitude', 0.0))
            self.gis_latitude_input.setValue(origin.get('latitude', 0.0))
            self.gis_altitude_input.setValue(origin.get('altitude', 0.0))
            self.gis_scale_input.setValue(config.get('scale', 1.0))
            self.gis_rotation_input.setValue(config.get('rotation', 0.0))
            
            self.log(f"GIS配置已加载: {config_path}", "info")
        except Exception as e:
            self.log(f"加载GIS配置失败: {e}", "error")
            QMessageBox.critical(self, "错误", f"加载GIS配置失败: {e}")
    
    def get_coordinate_transformer(self) -> CoordinateTransformer:
        """获取当前配置的坐标转换器"""
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer()
        
        # 更新转换器参数
        self.coordinate_transformer.set_origin(
            self.gis_longitude_input.value(),
            self.gis_latitude_input.value(),
            self.gis_altitude_input.value()
        )
        self.coordinate_transformer.set_scale(self.gis_scale_input.value())
        self.coordinate_transformer.set_rotation(self.gis_rotation_input.value())
        
        return self.coordinate_transformer
    
    def export_model_to_gis(self):
        """导出模型到GIS格式"""
        if not GIS_AVAILABLE or self.gaussians is None:
            QMessageBox.warning(self, "警告", "请先加载3DGS模型")
            return
        
        export_format = self.gis_export_format_combo.currentText()
        sample_rate = self.gis_sample_rate_slider.value() / 100.0
        opacity_threshold = self.gis_opacity_threshold_slider.value() / 100.0
        
        try:
            # 创建转换器
            transformer = self.get_coordinate_transformer()
            converter = GaussianSplattingToGIS(transformer)
            
            if export_format == "点云JSON":
                output_path, _ = QFileDialog.getSaveFileName(
                    self, "导出点云JSON", "", "JSON文件 (*.json)"
                )
                if not output_path:
                    return
                
                self.log("正在导出点云JSON...", "info")
                converter.convert_to_point_cloud_json(
                    self.gaussians,
                    output_path,
                    sample_rate=sample_rate,
                    opacity_threshold=opacity_threshold
                )
                self.log(f"点云JSON导出成功: {output_path}", "info")
                
            elif export_format == "3D Tiles":
                output_dir = QFileDialog.getExistingDirectory(
                    self, "选择3D Tiles输出目录"
                )
                if not output_dir:
                    return
                
                self.log("正在导出3D Tiles...", "info")
                tileset_path = converter.convert_to_3dtiles(
                    self.gaussians,
                    output_dir,
                    sample_rate=sample_rate,
                    opacity_threshold=opacity_threshold
                )
                self.log(f"3D Tiles导出成功: {tileset_path}", "info")
                
        except Exception as e:
            self.log(f"导出失败: {e}", "error")
            QMessageBox.critical(self, "错误", f"导出失败: {e}")
    
    def load_model_to_cesium(self):
        """加载3DGS模型到Cesium"""
        if not GIS_AVAILABLE or self.gaussians is None or self.cesium_widget is None:
            QMessageBox.warning(self, "警告", "请先加载3DGS模型和Cesium视图")
            return
        
        try:
            self.log("正在加载3DGS模型到Cesium...", "info")
            
            # 检查是否启用自动地理定位
            auto_geo = self.gis_auto_geo_checkbox.isChecked() if hasattr(self, 'gis_auto_geo_checkbox') else False
            
            if auto_geo:
                self.log("正在自动检测地理位置...", "info")
                
                # 获取图像目录 - 使用COLMAP输入路径
                image_directory = None
                
                # 优先使用COLMAP输入路径
                if hasattr(self, 'colmap_input_path_edit') and self.colmap_input_path_edit.text():
                    colmap_path = self.colmap_input_path_edit.text()
                    if os.path.exists(colmap_path):
                        image_directory = colmap_path
                        self.log(f"使用COLMAP输入路径: {image_directory}", "info")
                
                # 如果COLMAP路径不可用，回退到gaussians.source_path
                if not image_directory and hasattr(self.gaussians, 'source_path'):
                    source_path = self.gaussians.source_path
                    if os.path.isdir(source_path):
                        image_directory = source_path
                    else:
                        image_directory = os.path.dirname(source_path)
                    self.log(f"使用gaussians.source_path: {image_directory}", "info")
                
                # 使用模型管理器自动检测地理位置
                from cesium_model_manager import CesiumModelManager
                model_manager = CesiumModelManager(self.cesium_widget)
                
                # 添加调试信息
                self.log(f"正在检测图像目录: {image_directory}", "info")
                if image_directory and os.path.exists(image_directory):
                    self.log(f"图像目录存在，包含文件: {os.listdir(image_directory)[:5]}", "info")
                
                geo_location = model_manager.auto_detect_geo_location(image_directory, self.gaussians)
                
                if geo_location:
                    lon = geo_location['longitude']
                    lat = geo_location['latitude']
                    alt = geo_location['altitude']
                    
                    self.log(f"自动检测到地理位置: 经度={lon:.6f}, 纬度={lat:.6f}, 海拔={alt:.2f}m", "info")
                    
                    # 更新UI显示
                    self.gis_longitude_input.setValue(lon)
                    self.gis_latitude_input.setValue(lat)
                    self.gis_altitude_input.setValue(alt)
                else:
                    self.log("无法自动检测地理位置，使用手动设置的位置", "warning")
                    lon = float(self.gis_longitude_input.value())
                    lat = float(self.gis_latitude_input.value())
                    alt = float(self.gis_altitude_input.value())
            else:
                # 使用手动设置的位置
                lon = float(self.gis_longitude_input.value())
                lat = float(self.gis_latitude_input.value())
                alt = float(self.gis_altitude_input.value())
            
            # 导入PLY到splat转换器
            try:
                from ply_to_splat_converter import convert_gaussians_to_splat
            except ImportError:
                self.log("错误: 无法导入ply_to_splat_converter模块", "error")
                QMessageBox.critical(self, "错误", "无法导入PLY到splat转换器")
                return
            
            # 创建临时目录存放.splat文件
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='gsse_cesium_')
            splat_path = os.path.join(temp_dir, 'model.splat')
            
            # 转换为.splat格式
            self.log("正在转换模型为.splat格式...", "info")
            convert_gaussians_to_splat(self.gaussians, splat_path)
            
            # 获取缩放
            scale = float(self.gis_scale_input.value())
            
            # 获取旋转（度转弧度）
            rotation_deg = float(self.gis_rotation_input.value())
            rotation_z = np.radians(rotation_deg)
            
            # 使用HTTP服务器提供文件访问
            if self.http_server_started and HTTP_SERVER_AVAILABLE:
                 http_server = get_global_server()
                 if http_server:
                    # 生成唯一的文件名
                    import time
                    timestamp = int(time.time() * 1000)
                    filename = f"model_{timestamp}.splat"
                    splat_url = http_server.add_file(splat_path, filename)
                    self.log(f"已通过HTTP服务器提供文件: {splat_url}", "info")
                 else:
                    self.log("HTTP服务器未启动，无法加载模型到Cesium", "error")
                    return
            else:
                self.log("HTTP服务器不可用，无法加载模型到Cesium", "error")
                return
            
            self.log(f"模型已转换为.splat格式", "info")
            self.log(f"位置: ({lon:.6f}, {lat:.6f}, {alt:.2f})", "info")
            
            # 加载到Cesium
            self.cesium_widget.load_3dgs(
                splat_url,
                lon, lat, alt,
                scale=scale,
                rotation_x=0.0,
                rotation_y=0.0,
                rotation_z=rotation_z,
                fly_to=True
            )
            
            self.log("3DGS模型已发送到Cesium", "info")
            
        except Exception as e:
            import traceback
            self.log(f"加载到Cesium失败: {e}", "error")
            self.log(f"详细错误: {traceback.format_exc()}", "error")
            QMessageBox.critical(self, "错误", f"加载到Cesium失败: {e}")
    
    def load_local_model_to_cesium(self):
        """加载本地模型文件到Cesium"""
        if not GIS_AVAILABLE or self.cesium_widget is None:
            QMessageBox.warning(self, "警告", "请先加载Cesium视图")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "Gaussian Splat Files (*.ply *.splat *.ksplat);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            self.log(f"正在加载本地模型: {file_path}", "info")
            
            # 获取当前设置的位置
            lon = float(self.gis_longitude_input.value())
            lat = float(self.gis_latitude_input.value())
            alt = float(self.gis_altitude_input.value())
            scale = float(self.gis_scale_input.value())
            
            from cesium_model_manager import CesiumModelManager
            # 注意：这里应该复用已有的manager实例，但当前架构是在load_model_to_cesium中临时创建的
            # 最好将manager作为类成员
            if not hasattr(self, 'cesium_model_manager'):
                self.cesium_model_manager = CesiumModelManager(self.cesium_widget)
            
            model_id = self.cesium_model_manager.load_local_model(
                file_path, lon, lat, alt, scale=scale, fly_to=True
            )
            
            self.log(f"本地模型加载成功: {model_id}", "info")
            
        except Exception as e:
            import traceback
            self.log(f"加载本地模型失败: {e}", "error")
            self.log(f"详细错误: {traceback.format_exc()}", "error")
            QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def clear_cesium_scene(self):
        """清除Cesium场景"""
        if not GIS_AVAILABLE or self.cesium_widget is None:
            return
        
        self.cesium_widget.clear_all()
        self.log("已清除Cesium场景", "info")
    
    def on_cesium_viewer_ready(self):
        """Cesium viewer准备就绪回调"""
        self.log("Cesium viewer已就绪", "info")
        if GIS_AVAILABLE:
            self.gis_status_label.setText("Cesium viewer已就绪")
            # 启用相关按钮
            if self.model_loaded:
                self.gis_export_btn.setEnabled(True)
                self.gis_load_to_cesium_btn.setEnabled(True)
    
    def on_cesium_load_complete(self, success: bool, error: str):
        """Cesium加载完成回调"""
        if success:
            self.log("Cesium模型加载成功", "info")
            if GIS_AVAILABLE:
                self.gis_status_label.setText("模型加载成功")
        else:
            self.log(f"Cesium模型加载失败: {error}", "error")
            if GIS_AVAILABLE:
                self.gis_status_label.setText(f"加载失败: {error}")
    
    def on_cesium_object_clicked(self, data: dict):
        """Cesium对象点击回调"""
        position = data.get('position', {})
        lon = position.get('longitude', 0)
        lat = position.get('latitude', 0)
        height = position.get('height', 0)
        self.log(f"点击位置: 经度={lon:.6f}°, 纬度={lat:.6f}°, 高度={height:.2f}m", "info")


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("GSSE")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口（主题会在窗口初始化时应用）
    window = GSSEGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


