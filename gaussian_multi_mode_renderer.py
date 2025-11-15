#!/usr/bin/env python3
"""
多模式3DGS渲染器
支持多种渲染模式：点云、椭球体、深度、法线、标准Splat等
基于pyqtgraph.opengl实现高性能渲染
"""

import numpy as np
import torch
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QSlider, QDoubleSpinBox, QGroupBox, QComboBox,
                              QCheckBox, QSpinBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QEvent, QRect
from PyQt5.QtGui import QMouseEvent, QWheelEvent, QPainter, QPen, QColor, QBrush
from types import SimpleNamespace

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False

# 导入Open3D用于mesh生成
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except Exception:
    OPEN3D_AVAILABLE = False



class RenderMode(Enum):
    """渲染模式枚举（参考DIVSHOT）"""
    SPLAT = 0           # 标准高斯Splat
    POINT_CLOUD = 1     # 点云模式
    DEPTH = 2           # 深度可视化
    NORMAL = 3          # 法线可视化
    RINGS = 4           # 环形可视化
    ELLIPSOIDS = 5      # 椭球体可视化
    CENTERS = 6         # 中心点可视化
    MESH = 7            # Mesh网格可视化


@dataclass
class RenderSettings:
    """渲染设置（参考DIVSHOT的RenderSettings）"""
    render_mode: RenderMode = RenderMode.POINT_CLOUD
    point_size: float = 2.0
    ellipsoid_scale: float = 1.0
    depth_range: Tuple[float, float] = (0.0, 10.0)
    show_wireframe: bool = False
    alpha: float = 1.0
    color_map: str = 'viridis'  # 用于深度可视化
    lod_enabled: bool = True    # LOD（细节层次）
    max_points: int = 500000    # 最大显示点数
    use_gpu_accel: bool = True  # GPU加速
    splat_scaling_modifier: float = 0.2  # SPLAT半径缩放
    # 坐标系设置（已固定为COLMAP→pyqtgraph，无需UI）


class EllipsoidRenderer:
    """椭球体渲染器"""
    
    def __init__(self, resolution: int = 8):
        """
        初始化椭球体渲染器
        
        Args:
            resolution: 椭球体网格分辨率（越高越平滑，但更耗性能）
        """
        self.resolution = resolution
        self.sphere_mesh = self._create_unit_sphere_mesh(resolution)
    
    def _create_unit_sphere_mesh(self, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建单位球体网格
        
        Returns:
            vertices: [N, 3] 顶点数组
            faces: [M, 3] 面片索引
        """
        # 使用UV球体参数化
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        
        # 创建网格
        U, V = np.meshgrid(u, v)
        
        # 球坐标转笛卡尔坐标
        x = np.sin(V) * np.cos(U)
        y = np.sin(V) * np.sin(U)
        z = np.cos(V)
        
        # 展平
        vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # 创建面片索引
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # 每个四边形分成两个三角形
                v1 = i * resolution + j
                v2 = v1 + 1
                v3 = (i + 1) * resolution + j
                v4 = v3 + 1
                
                faces.append([v1, v3, v2])
                faces.append([v2, v3, v4])
        
        faces = np.array(faces, dtype=np.uint32)
        
        return vertices, faces
    
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        四元数转旋转矩阵
        
        Args:
            q: [N, 4] 四元数数组 (w, x, y, z)
            
        Returns:
            R: [N, 3, 3] 旋转矩阵
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # 构建旋转矩阵
        R = np.zeros((len(q), 3, 3))
        
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        
        return R
    
    def create_ellipsoid_meshes(self, 
                                positions: np.ndarray,
                                rotations: np.ndarray,
                                scales: np.ndarray,
                                colors: np.ndarray,
                                max_ellipsoids: int = 1000) -> list:
        """
        创建椭球体网格列表
        
        Args:
            positions: [N, 3] 位置
            rotations: [N, 4] 旋转四元数
            scales: [N, 3] 缩放
            colors: [N, 3] 颜色
            max_ellipsoids: 最大椭球体数量（性能限制）
            
        Returns:
            mesh_items: GL mesh项列表
        """
        if not PG_AVAILABLE:
            return []
        
        # 限制数量
        n = min(len(positions), max_ellipsoids)
        if n < len(positions):
            # 随机采样
            indices = np.random.choice(len(positions), n, replace=False)
            positions = positions[indices]
            rotations = rotations[indices]
            scales = scales[indices]
            colors = colors[indices]
        
        mesh_items = []
        
        # 获取单位球面顶点和面片
        unit_verts, faces = self.sphere_mesh
        
        # 转换旋转矩阵
        R = self.quaternion_to_rotation_matrix(rotations)
        
        # 为每个椭球体创建mesh
        for i in range(n):
            # 变换顶点：先缩放，再旋转，再平移
            # V' = R * S * V + T
            S = np.diag(scales[i])
            transformed_verts = (R[i] @ S @ unit_verts.T).T + positions[i]
            
            # 创建mesh数据
            mesh_data = gl.MeshData(vertexes=transformed_verts, faces=faces)
            
            # 创建mesh item
            color = tuple(colors[i]) + (0.6,)  # 添加透明度
            mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                color=color,
                shader='balloon',  # 使用气球shader使其更平滑
                smooth=True,
                drawEdges=False,
                glOptions='translucent'
            )
            
            mesh_items.append(mesh_item)
        
        return mesh_items
    
    def create_ellipsoid_wireframes(self,
                                   positions: np.ndarray,
                                   rotations: np.ndarray,
                                   scales: np.ndarray,
                                   colors: np.ndarray,
                                   max_ellipsoids: int = 5000) -> list:
        """
        创建椭球体线框（性能更好）
        
        Args:
            positions: [N, 3] 位置
            rotations: [N, 4] 旋转四元数
            scales: [N, 3] 缩放
            colors: [N, 3] 颜色
            max_ellipsoids: 最大椭球体数量
            
        Returns:
            line_items: GL线框项列表
        """
        if not PG_AVAILABLE:
            return []
        
        # 限制数量
        n = min(len(positions), max_ellipsoids)
        if n < len(positions):
            indices = np.random.choice(len(positions), n, replace=False)
            positions = positions[indices]
            rotations = rotations[indices]
            scales = scales[indices]
            colors = colors[indices]
        
        line_items = []
        
        # 创建单位圆环（3个正交平面）
        theta = np.linspace(0, 2*np.pi, 32)
        circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        
        # 三个主平面的圆环
        circles = [
            circle,  # XY平面
            np.column_stack([circle[:, 0], np.zeros_like(theta), circle[:, 1]]),  # XZ平面
            np.column_stack([np.zeros_like(theta), circle[:, 0], circle[:, 1]])   # YZ平面
        ]
        
        # 转换旋转矩阵
        R = self.quaternion_to_rotation_matrix(rotations)
        
        for i in range(n):
            S = np.diag(scales[i])
            
            for circle_points in circles:
                # 变换圆环
                transformed = (R[i] @ S @ circle_points.T).T + positions[i]
                
                # 创建线条
                color = tuple(colors[i]) + (0.8,)
                line_item = gl.GLLinePlotItem(
                    pos=transformed,
                    color=color,
                    width=1.5,
                    antialias=True
                )
                line_items.append(line_item)
        
        return line_items


class BatchedEllipsoidRenderer:
    """批量椭球体渲染器（更高性能）"""
    
    def __init__(self, resolution: int = 6):
        """
        初始化批量椭球体渲染器
        
        Args:
            resolution: 椭球体网格分辨率
        """
        self.resolution = resolution
        self.ellipsoid_renderer = EllipsoidRenderer(resolution)
    
    def create_instanced_ellipsoids(self,
                                   positions: np.ndarray,
                                   rotations: np.ndarray,
                                   scales: np.ndarray,
                                   colors: np.ndarray,
                                   max_instances: int = 10000) -> Optional[gl.GLScatterPlotItem]:
        """
        使用实例化渲染创建椭球体（高性能）
        
        注意：pyqtgraph的GLScatterPlotItem不直接支持实例化渲染
        这里使用点云近似，但可以通过自定义shader改进
        
        Args:
            positions: [N, 3] 位置
            rotations: [N, 4] 旋转四元数
            scales: [N, 3] 缩放
            colors: [N, 3] 颜色
            max_instances: 最大实例数量
            
        Returns:
            scatter_item: GLScatterPlotItem（使用大小变化的点近似椭球体）
        """
        if not PG_AVAILABLE:
            return None
        
        # 限制数量
        n = min(len(positions), max_instances)
        if n < len(positions):
            indices = np.random.choice(len(positions), n, replace=False)
            positions = positions[indices]
            scales = scales[indices]
            colors = colors[indices]
        
        # 计算椭球体的"有效大小"（平均半径）
        avg_scales = np.mean(scales, axis=1) * 10.0
        avg_scales = np.clip(avg_scales, 1.0, 20.0)
        
        # 添加透明度
        if colors.shape[1] == 3:
            colors = np.column_stack([colors, np.ones(n) * 0.5])
        
        # 创建scatter plot（用变化的点大小近似椭球体）
        scatter_item = gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=avg_scales,
            pxMode=True
        )
        
        return scatter_item
    
    def render_ellipsoids_lod(self,
                             positions: np.ndarray,
                             rotations: np.ndarray,
                             scales: np.ndarray,
                             colors: np.ndarray,
                             camera_distance: float = 10.0) -> list:
        """
        使用LOD（细节层次）渲染椭球体
        
        Args:
            positions: [N, 3] 位置
            rotations: [N, 4] 旋转四元数
            scales: [N, 3] 缩放
            colors: [N, 3] 颜色
            camera_distance: 相机距离
            
        Returns:
            render_items: 渲染项列表
        """
        # 根据距离选择渲染策略
        # 近距离：完整mesh
        # 中距离：线框
        # 远距离：点
        
        items = []
        
        n = len(positions)
        
        # 计算到相机的距离（简化：使用位置的模长）
        distances = np.linalg.norm(positions, axis=1)
        
        # LOD阈值
        close_threshold = camera_distance * 0.3
        medium_threshold = camera_distance * 1.0
        
        # 近距离：完整mesh（限制数量）
        close_mask = distances < close_threshold
        if np.any(close_mask) and np.sum(close_mask) < 100:
            close_items = self.ellipsoid_renderer.create_ellipsoid_meshes(
                positions[close_mask],
                rotations[close_mask],
                scales[close_mask],
                colors[close_mask],
                max_ellipsoids=100
            )
            items.extend(close_items)
        
        # 中距离：线框
        medium_mask = (distances >= close_threshold) & (distances < medium_threshold)
        if np.any(medium_mask):
            medium_items = self.ellipsoid_renderer.create_ellipsoid_wireframes(
                positions[medium_mask],
                rotations[medium_mask],
                scales[medium_mask],
                colors[medium_mask],
                max_ellipsoids=1000
            )
            items.extend(medium_items)
        
        # 远距离：点云
        far_mask = distances >= medium_threshold
        if np.any(far_mask):
            far_item = self.create_instanced_ellipsoids(
                positions[far_mask],
                rotations[far_mask],
                scales[far_mask],
                colors[far_mask],
                max_instances=50000
            )
            if far_item:
                items.append(far_item)
        
        return items


class OrientationIndicator(QWidget):
    """3D坐标轴方向指示器（悬浮显示）"""
    
    def __init__(self, parent=None, size=100):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 鼠标事件穿透
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)  # 无边框且置顶
        
        self.size = size
        self.setFixedSize(size, size)
        
        # 旋转矩阵（从父视图同步）
        self.rotation = np.eye(3)
        
        # 坐标轴颜色
        self.axis_colors = {
            'X': QColor(255, 80, 80, 220),    # 红色
            'Y': QColor(80, 255, 80, 220),    # 绿色
            'Z': QColor(80, 120, 255, 220)    # 蓝色
        }
        
    def set_rotation(self, view_matrix):
        """从视图矩阵提取旋转"""
        if view_matrix is not None:
            # 提取旋转部分（3x3左上角）
            self.rotation = view_matrix[:3, :3].copy()
            self.update()
    
    def paintEvent(self, event):
        """绘制坐标轴指示器"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 绘制半透明背景圆形
        center = self.size / 2
        radius = self.size * 0.45
        
        # 背景圆
        painter.setBrush(QBrush(QColor(40, 40, 40, 150)))
        painter.setPen(QPen(QColor(80, 80, 80, 180), 2))
        painter.drawEllipse(int(center - radius), int(center - radius), 
                          int(2 * radius), int(2 * radius))
        
        # 定义3D坐标轴方向
        axes_3d = {
            'X': np.array([1, 0, 0]),
            'Y': np.array([0, 1, 0]),
            'Z': np.array([0, 0, 1])
        }
        
        # 应用旋转并投影到2D
        axes_2d = {}
        axes_depth = {}
        for name, axis_3d in axes_3d.items():
            # 旋转
            rotated = self.rotation @ axis_3d
            # 2D投影（忽略深度）- 注意Y轴翻转
            axes_2d[name] = np.array([rotated[0], -rotated[1]]) * radius * 0.7
            axes_depth[name] = rotated[2]  # 保存深度用于排序
        
        # 按深度排序绘制（先绘制远的，后绘制近的）
        sorted_axes = sorted(axes_3d.keys(), key=lambda k: axes_depth[k])
        
        # 绘制坐标轴
        for name in sorted_axes:
            pos_2d = axes_2d[name]
            depth = axes_depth[name]
            
            # 根据深度调整透明度
            alpha = 255 if depth > 0 else 150
            color = self.axis_colors[name]
            color.setAlpha(alpha)
            
            # 绘制轴线
            line_width = 3 if depth > 0 else 2
            painter.setPen(QPen(color, line_width, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(int(center), int(center),
                           int(center + pos_2d[0]), int(center + pos_2d[1]))
            
            # 绘制箭头
            end_x = center + pos_2d[0]
            end_y = center + pos_2d[1]
            
            # 计算箭头方向
            angle = np.arctan2(pos_2d[1], pos_2d[0])
            arrow_size = 8 if depth > 0 else 6
            
            # 箭头的两个翼
            arrow_angle = np.pi / 6  # 30度
            left_angle = angle + np.pi - arrow_angle
            right_angle = angle + np.pi + arrow_angle
            
            left_x = end_x + arrow_size * np.cos(left_angle)
            left_y = end_y + arrow_size * np.sin(left_angle)
            right_x = end_x + arrow_size * np.cos(right_angle)
            right_y = end_y + arrow_size * np.sin(right_angle)
            
            # 绘制箭头（填充三角形）
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            from PyQt5.QtGui import QPolygonF
            from PyQt5.QtCore import QPointF
            arrow = QPolygonF([
                QPointF(end_x, end_y),
                QPointF(left_x, left_y),
                QPointF(right_x, right_y)
            ])
            painter.drawPolygon(arrow)
            
            # 绘制标签
            label_offset = 15
            label_x = center + pos_2d[0] * 1.3
            label_y = center + pos_2d[1] * 1.3
            
            painter.setPen(QPen(color, 1))
            painter.setFont(painter.font())
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            
            # 文字居中绘制
            text_rect = painter.fontMetrics().boundingRect(name)
            painter.drawText(int(label_x - text_rect.width() / 2),
                           int(label_y + text_rect.height() / 4),
                           name)
        
        painter.end()


class SelectionOverlay(QWidget):
    """透明覆盖层用于绘制选择框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # 鼠标事件穿透
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        
        self.selection_rect = None
        self.selection_path = []
        self.selection_type = 'box'
        
    def set_selection(self, start, end, sel_type='box', path=None):
        """设置选择区域"""
        self.selection_type = sel_type
        if start and end:
            self.selection_rect = QRect(
                min(start.x(), end.x()),
                min(start.y(), end.y()),
                abs(end.x() - start.x()),
                abs(end.y() - start.y())
            )
        else:
            self.selection_rect = None
        
        if path:
            self.selection_path = path
        else:
            self.selection_path = []
        
        self.update()
    
    def clear_selection(self):
        """清除选择"""
        self.selection_rect = None
        self.selection_path = []
        self.update()
    
    def paintEvent(self, event):
        """绘制选择框"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.selection_rect and self.selection_type == 'box':
            # 绘制矩形框
            pen = QPen(QColor(0, 150, 255, 220), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 150, 255, 40)))
            painter.drawRect(self.selection_rect)
            
        elif self.selection_path and self.selection_type in ['lasso', 'brush']:
            # 绘制路径
            pen = QPen(QColor(0, 150, 255, 220), 2, Qt.SolidLine)
            painter.setPen(pen)
            if len(self.selection_path) > 1:
                for i in range(len(self.selection_path) - 1):
                    painter.drawLine(self.selection_path[i], self.selection_path[i + 1])
                # 如果是套索，连接首尾
                if self.selection_type == 'lasso' and len(self.selection_path) > 2:
                    painter.drawLine(self.selection_path[-1], self.selection_path[0])
        
        painter.end()
    
    def resizeEvent(self, event):
        """窗口大小改变时更新"""
        super().resizeEvent(event)
        # 确保覆盖层覆盖整个父widget
        if self.parent():
            self.setGeometry(0, 0, self.parent().width(), self.parent().height())


class GaussianMultiModeRenderer(QWidget):
    """多模式3DGS渲染器（高性能）"""
    
    selection_changed = pyqtSignal()
    mode_changed = pyqtSignal(RenderMode)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        if not PG_AVAILABLE:
            raise RuntimeError("pyqtgraph 未安装，无法使用GL 3D视图")
        
        self.gaussians = None
        self.editor = None
        self.settings = RenderSettings()
        
        # 固定坐标系矩阵：COLMAP(x右,y下,z前) → pyqtgraph(x右,y后,z上)
        self._coord_matrix = np.array([[1.0, 0.0, 0.0],
                                       [0.0, 0.0, -1.0],
                                       [0.0, -1.0, 0.0]], dtype=np.float32)
        
        # 模型中心偏移（用于将模型移至坐标轴原点）
        self.model_center = np.zeros(3, dtype=np.float32)

        # 渲染项
        self.render_items = {}  # {mode: gl_item}
        self.current_item = None
        self.current_items_list = []  # 用于多个渲染项
        
        # Mesh缓存
        self.cached_mesh = None  # 缓存生成的mesh
        
        # 相机参数
        self.camera_view = None
        self.camera_proj = None
        
        # 选择状态
        self.selection_mode = False
        self.selection_type = 'box'  # 'box', 'lasso', 'brush'
        self.is_selecting = False
        self.select_start = None
        self.select_end = None
        self.select_path = []  # 用于套索/笔刷
        
        # 椭球体渲染器（内置支持）
        self.ellipsoid_renderer = EllipsoidRenderer(resolution=8)
        self.batched_ellipsoid_renderer = BatchedEllipsoidRenderer(resolution=6)
        
        # 性能统计
        self.frame_count = 0
        self.last_update_time = 0
        
        # 保存初始相机参数（完整版）
        self.initial_camera_params = {
            'distance': 10,
            'elevation': 30,
            'azimuth': 45,
            'center': None,  # 将在init_ui后设置
            'rotation': None,  # 旋转状态
            'fov': 60  # 视场角
        }
        
        self.init_ui()
        self.setup_view()
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建一个容器来放置GL视图和覆盖层
        self.view_container = QWidget()
        view_container_layout = QVBoxLayout(self.view_container)
        view_container_layout.setContentsMargins(0, 0, 0, 0)
        view_container_layout.setSpacing(0)
        
        # 创建GL视图
        self.view = gl.GLViewWidget()
        self.view.opts['rotationMethod'] = 'quaternion'  # 先设置旋转方法
        # 使用API设置初始相机，而不是直接改opts，确保在四元数模式下生效
        self.view.setCameraPosition(
            distance=self.initial_camera_params['distance'],
            elevation=self.initial_camera_params['elevation'],
            azimuth=self.initial_camera_params['azimuth']
        )
        # 保存完整的初始相机状态（pyqtgraph的Vector对象需要转换）
        from pyqtgraph import Vector
        from pyqtgraph.Qt import QtGui
        
        center = self.view.opts['center']
        if isinstance(center, Vector):
            self.initial_camera_params['center'] = Vector(center)
        else:
            self.initial_camera_params['center'] = np.array(center).copy()
        
        # 保存初始旋转（四元数）
        if 'rotation' in self.view.opts and self.view.opts['rotation'] is not None:
            rot = self.view.opts['rotation']
            if hasattr(rot, 'copy'):
                self.initial_camera_params['rotation'] = rot.copy()
            else:
                # 如果是QtGui.QQuaternion，需要创建新实例
                self.initial_camera_params['rotation'] = QtGui.QQuaternion(rot)
        
        view_container_layout.addWidget(self.view)
        
        # 创建透明覆盖层用于绘制选择框
        self.overlay = SelectionOverlay(self.view_container)
        self.overlay.show()
        self.overlay.raise_()  # 确保在最上层
        
        # 创建3D坐标轴方向指示器
        self.orientation_indicator = OrientationIndicator(self.view_container, size=100)
        self.orientation_indicator.show()
        self.orientation_indicator.raise_()  # 确保在最上层
        self._position_orientation_indicator()
        
        # SPLAT渲染结果覆盖层（图像，不拦截鼠标）
        self.splat_label = QLabel(self.view_container)
        self.splat_label.setScaledContents(True)
        self.splat_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.splat_label.hide()
        
        main_layout.addWidget(self.view_container, stretch=1)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QGroupBox("渲染控制")
        layout = QVBoxLayout(panel)
        
        # 第一行：渲染模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("渲染模式:"))
        
        self.mode_combo = QComboBox()
        for mode in RenderMode:
            self.mode_combo.addItem(mode.name, mode)
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        # 与默认渲染模式同步下拉框初始值，避免UI与实际不一致
        try:
            index = self.mode_combo.findData(self.settings.render_mode)
            if index >= 0:
                self.mode_combo.blockSignals(True)
                self.mode_combo.setCurrentIndex(index)
                self.mode_combo.blockSignals(False)
        except Exception:
            pass
        
        # 快捷切换按钮
        self.btn_point = QPushButton("点云")
        self.btn_point.clicked.connect(lambda: self.set_render_mode(RenderMode.POINT_CLOUD))
        mode_layout.addWidget(self.btn_point)
        
        self.btn_ellipsoid = QPushButton("椭球体")
        self.btn_ellipsoid.clicked.connect(lambda: self.set_render_mode(RenderMode.ELLIPSOIDS))
        mode_layout.addWidget(self.btn_ellipsoid)
        
        self.btn_depth = QPushButton("深度")
        self.btn_depth.clicked.connect(lambda: self.set_render_mode(RenderMode.DEPTH))
        mode_layout.addWidget(self.btn_depth)
        
        self.btn_splat = QPushButton("Splat")
        self.btn_splat.clicked.connect(lambda: self.set_render_mode(RenderMode.SPLAT))
        mode_layout.addWidget(self.btn_splat)
        
        self.btn_mesh = QPushButton("Mesh")
        self.btn_mesh.clicked.connect(self.on_mesh_button_clicked)
        self.btn_mesh.setEnabled(False)  # 默认禁用，直到生成mesh
        self.btn_mesh.setToolTip("请先通过菜单生成mesh (Ctrl+M)")
        mode_layout.addWidget(self.btn_mesh)
        
        # 视角复位按钮
        self.btn_reset_view = QPushButton("复位视角")
        self.btn_reset_view.clicked.connect(self.reset_camera_view)
        mode_layout.addWidget(self.btn_reset_view)
        
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # 第二行：渲染参数
        param_layout = QHBoxLayout()
        
        # 点大小
        param_layout.addWidget(QLabel("点大小:"))
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.1, 20.0)
        self.point_size_spin.setValue(self.settings.point_size)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)
        param_layout.addWidget(self.point_size_spin)
        
        # 透明度
        param_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.settings.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        param_layout.addWidget(self.alpha_slider)
        
        # LOD开关
        self.lod_checkbox = QCheckBox("LOD优化")
        self.lod_checkbox.setChecked(self.settings.lod_enabled)
        self.lod_checkbox.stateChanged.connect(self.on_lod_changed)
        param_layout.addWidget(self.lod_checkbox)
        
        # 最大点数
        param_layout.addWidget(QLabel("最大点数:"))
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(10000, 2000000)
        self.max_points_spin.setSingleStep(50000)
        self.max_points_spin.setValue(self.settings.max_points)
        self.max_points_spin.valueChanged.connect(self.on_max_points_changed)
        param_layout.addWidget(self.max_points_spin)
        
        param_layout.addStretch()
        layout.addLayout(param_layout)
        
        # 统计信息
        self.stats_label = QLabel("就绪")
        layout.addWidget(self.stats_label)
        
        return panel
    
    def _position_orientation_indicator(self):
        """将方向指示器定位到右上角"""
        if hasattr(self, 'orientation_indicator') and hasattr(self, 'view_container'):
            margin = 10
            size = self.orientation_indicator.size
            width = self.view_container.width()
            x = width - size - margin
            y = margin
            self.orientation_indicator.move(x, y)
    
    def setup_view(self):
        """设置视图"""
        
        # 保存原始鼠标事件处理（在view创建后）
        self._original_mouse_press = self.view.mousePressEvent
        self._original_mouse_move = self.view.mouseMoveEvent
        self._original_mouse_release = self.view.mouseReleaseEvent
        
        # 重写鼠标事件
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_move
        self.view.mouseReleaseEvent = self.on_mouse_release
        
        # 设置widget属性以启用透明背景绘制
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
    
    def reset_camera_view(self):
        """复位相机视角到初始状态"""
        try:
            from pyqtgraph import Vector
            from pyqtgraph.Qt import QtGui
            
            # 方法1: 直接重置opts（最彻底）
            # 1. 复位center位置
            if self.initial_camera_params['center'] is not None:
                saved_center = self.initial_camera_params['center']
                if isinstance(saved_center, Vector):
                    self.view.opts['center'] = Vector(saved_center)
                elif isinstance(saved_center, np.ndarray):
                    self.view.opts['center'] = Vector(saved_center)
                else:
                    self.view.opts['center'] = saved_center
            
            # 2. 复位旋转（四元数）- 这是关键！
            if self.initial_camera_params['rotation'] is not None:
                saved_rotation = self.initial_camera_params['rotation']
                if isinstance(saved_rotation, QtGui.QQuaternion):
                    self.view.opts['rotation'] = QtGui.QQuaternion(saved_rotation)
                else:
                    self.view.opts['rotation'] = saved_rotation
            
            # 3. 复位距离
            self.view.opts['distance'] = self.initial_camera_params['distance']
            
            # 4. 复位elevation和azimuth（如果不使用四元数）
            if 'elevation' in self.view.opts:
                self.view.opts['elevation'] = self.initial_camera_params['elevation']
            if 'azimuth' in self.view.opts:
                self.view.opts['azimuth'] = self.initial_camera_params['azimuth']
            
            # 5. 重置视场角
            self.view.opts['fov'] = self.initial_camera_params['fov']
            
            # 6. 强制更新视图
            self.view.update()
            
            # 7. 更新相机参数
            self._update_camera_params()
            
            self.log("视角已复位", "info")
        except Exception as e:
            import traceback
            self.log(f"复位视角失败: {e}\n{traceback.format_exc()}", "warning")
    
    def set_gaussians(self, gaussians, editor=None):
        """设置Gaussian模型"""
        self.gaussians = gaussians
        self.editor = editor
        
        # 计算并保存模型中心偏移
        if self.gaussians is not None:
            try:
                xyz = self.gaussians.get_xyz.detach()
                self.model_center = xyz.mean(dim=0).cpu().numpy()
                self.log(f"模型中心: [{self.model_center[0]:.3f}, {self.model_center[1]:.3f}, {self.model_center[2]:.3f}]")
            except Exception:
                self.model_center = np.zeros(3, dtype=np.float32)
        else:
            self.model_center = np.zeros(3, dtype=np.float32)
        
        self.update_view()
    
    def set_render_mode(self, mode: RenderMode):
        """设置渲染模式"""
        if self.settings.render_mode == mode:
            return
        
        self.settings.render_mode = mode
        
        # 更新UI
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        
        # 更新渲染
        self.update_view()
        self.mode_changed.emit(mode)
        
        self.log(f"渲染模式切换至: {mode.name}")
    
    def on_mode_changed(self, index):
        """模式切换回调"""
        mode = self.mode_combo.itemData(index)
        if mode:
            # 如果切换到MESH模式但没有生成过mesh，给出提示
            if mode == RenderMode.MESH and self.cached_mesh is None:
                self.log("请先通过菜单生成mesh (Ctrl+M)", "warning")
                # 恢复到之前的模式
                prev_index = self.mode_combo.findData(self.settings.render_mode)
                if prev_index >= 0:
                    self.mode_combo.blockSignals(True)
                    self.mode_combo.setCurrentIndex(prev_index)
                    self.mode_combo.blockSignals(False)
                return
            
            self.set_render_mode(mode)
    
    def on_mesh_button_clicked(self):
        """Mesh按钮点击处理"""
        if self.cached_mesh is None:
            self.log("请先通过菜单生成mesh (Ctrl+M)", "warning")
            return
        self.set_render_mode(RenderMode.MESH)
    
    def on_point_size_changed(self, value):
        """点大小改变"""
        self.settings.point_size = value
        self.update_view()
    
    def on_alpha_changed(self, value):
        """透明度改变"""
        self.settings.alpha = value / 100.0
        self.update_view()
    
    def on_lod_changed(self, state):
        """LOD开关"""
        self.settings.lod_enabled = (state == Qt.Checked)
        self.update_view()
    
    def on_max_points_changed(self, value):
        """最大点数改变"""
        self.settings.max_points = value
        self.update_view()
    
    def get_render_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """获取渲染数据（位置、颜色、索引）"""
        if self.gaussians is None:
            return None, None, []
        
        # 获取可见点
        if self.editor:
            visible_indices = self.editor.get_visible_indices()
        else:
            visible_indices = torch.arange(self.gaussians.get_xyz.shape[0], 
                                          device=self.gaussians.get_xyz.device)
        
        if len(visible_indices) == 0:
            return None, None, []
        
        # LOD下采样
        if self.settings.lod_enabled and len(visible_indices) > self.settings.max_points:
            # 使用GPU加速的随机采样
            sample_indices = torch.randperm(len(visible_indices), 
                                          device=visible_indices.device)[:self.settings.max_points]
            visible_indices = visible_indices[sample_indices]
        
        # 获取位置并应用坐标变换与中心化
        positions = self.gaussians.get_xyz[visible_indices].detach().cpu().numpy()
        positions = positions - self.model_center  # 平移到原点
        positions = self._apply_coord_transform_np(positions)
        
        # 根据渲染模式计算颜色
        colors = self.compute_colors_for_mode(visible_indices)
        
        return positions, colors, visible_indices
    
    def compute_colors_for_mode(self, indices) -> np.ndarray:
        """根据渲染模式计算颜色"""
        mode = self.settings.render_mode
        
        if mode == RenderMode.POINT_CLOUD or mode == RenderMode.SPLAT:
            # 使用原始颜色（从features_dc提取）
            return self._get_gaussian_colors(indices)
        
        elif mode == RenderMode.DEPTH:
            # 深度可视化
            return self._compute_depth_colors(indices)
        
        elif mode == RenderMode.NORMAL:
            # 法线可视化
            return self._compute_normal_colors(indices)
        
        elif mode == RenderMode.ELLIPSOIDS or mode == RenderMode.RINGS:
            # 椭球体/环形使用原始颜色
            return self._get_gaussian_colors(indices)
        
        elif mode == RenderMode.CENTERS:
            # 中心点用单一颜色
            n = len(indices)
            return np.ones((n, 3), dtype=np.float32) * 0.8
        
        elif mode == RenderMode.MESH:
            # Mesh使用原始颜色
            return self._get_gaussian_colors(indices)
        
        else:
            # 默认灰色
            n = len(indices)
            return np.ones((n, 3), dtype=np.float32) * 0.7
    
    def _get_gaussian_colors(self, indices) -> np.ndarray:
        """获取高斯原始颜色"""
        try:
            fdc = self.gaussians._features_dc[indices].detach().cpu().numpy()
            if fdc.ndim == 3:
                if fdc.shape[1] == 1:
                    colors = fdc[:, 0, :]
                else:
                    colors = fdc[:, :, 0]
            else:
                colors = fdc
            # 从[-1,1]映射到[0,1]
            colors = np.clip(colors * 0.5 + 0.5, 0.0, 1.0)
        except Exception as e:
            # 如果获取颜色失败，使用默认颜色
            n = len(indices)
            colors = np.ones((n, 3), dtype=np.float32) * 0.7
        
        return colors.astype(np.float32)
    
    def _compute_depth_colors(self, indices) -> np.ndarray:
        """计算深度颜色（使用colormap）"""
        positions = self.gaussians.get_xyz[indices].detach().cpu().numpy()
        positions = positions - self.model_center  # 平移到原点
        positions = self._apply_coord_transform_np(positions)
        
        # 计算深度（相对于相机）
        # 这里使用z坐标作为深度
        depths = positions[:, 2]
        
        # 归一化到[0, 1]
        d_min, d_max = self.settings.depth_range
        depths_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
        
        # 应用colormap
        colors = self._apply_colormap(depths_norm, self.settings.color_map)
        
        return colors
    
    def _compute_normal_colors(self, indices) -> np.ndarray:
        """计算法线颜色（从协方差矩阵提取）"""
        try:
            # 尝试从高斯的旋转和缩放参数计算法线
            # 这里简化处理：使用主轴方向作为法线
            rotations = self.gaussians.get_rotation[indices].detach().cpu().numpy()
            
            # 四元数转法线（使用第一个主轴）
            # 简化：将四元数归一化后映射到RGB
            normals = np.abs(rotations[:, :3])
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
            
            # 映射到[0,1]
            colors = normals * 0.5 + 0.5
        except Exception:
            # 如果失败，返回固定颜色
            n = len(indices)
            colors = np.ones((n, 3), dtype=np.float32) * 0.5
        
        return colors.astype(np.float32)
    
    def _apply_colormap(self, values: np.ndarray, cmap_name: str) -> np.ndarray:
        """应用colormap"""
        try:
            from matplotlib import cm
            cmap = cm.get_cmap(cmap_name)
            colors = cmap(values)[:, :3]  # 只取RGB，不要alpha
            return colors.astype(np.float32)
        except Exception:
            # 如果matplotlib不可用，使用简单的灰度
            colors = np.stack([values, values, values], axis=1)
            return colors.astype(np.float32)
    
    def update_view(self):
        """更新视图"""
        if self.gaussians is None:
            return
        
        # 获取渲染数据
        positions, colors, indices = self.get_render_data()
        
        if positions is None or len(positions) == 0:
            self.stats_label.setText("无数据")
            return
        
        # 清除旧的渲染项（但保留sel_scatter和lock_scatter）
        if self.current_item is not None:
            self.view.removeItem(self.current_item)
            self.current_item = None
        
        # 清除items列表时保留sel_scatter和lock_scatter
        sel_scatter_backup = None
        lock_scatter_backup = None
        # 备份sel_scatter（无论是否在列表中）
        if hasattr(self, 'sel_scatter') and self.sel_scatter is not None:
            sel_scatter_backup = self.sel_scatter
        # 备份lock_scatter（无论是否在列表中）
        if hasattr(self, 'lock_scatter') and self.lock_scatter is not None:
            lock_scatter_backup = self.lock_scatter
        
        # 清除所有渲染项，但不要删除sel_scatter和lock_scatter
        for item in self.current_items_list:
            if item != sel_scatter_backup and item != lock_scatter_backup:
                self.view.removeItem(item)
        self.current_items_list = []
        
        # 恢复sel_scatter和lock_scatter到列表中（如果它们存在）
        if sel_scatter_backup is not None:
            self.current_items_list.append(sel_scatter_backup)
            # 确保sel_scatter在视图中（可能之前被移除了）
            if sel_scatter_backup not in self.view.items:
                self.view.addItem(sel_scatter_backup)
        if lock_scatter_backup is not None:
            self.current_items_list.append(lock_scatter_backup)
            # 确保lock_scatter在视图中（可能之前被移除了）
            if lock_scatter_backup not in self.view.items:
                self.view.addItem(lock_scatter_backup)
        
        # 根据模式创建渲染项
        mode = self.settings.render_mode
        # 非SPLAT模式隐藏图像覆盖层
        if mode != RenderMode.SPLAT and hasattr(self, 'splat_label'):
            try:
                self.splat_label.hide()
            except Exception:
                pass
        
        if mode == RenderMode.POINT_CLOUD or mode == RenderMode.CENTERS:
            self.current_item = self._render_as_points(positions, colors)
        
        elif mode == RenderMode.ELLIPSOIDS:
            self.current_item = self._render_as_ellipsoids(positions, colors, indices)
        
        elif mode == RenderMode.SPLAT:
            # 使用3DGS离屏渲染到叠加图像，保证与点云对齐
            self._render_as_splat_image()
            self.current_item = None
        
        elif mode == RenderMode.DEPTH or mode == RenderMode.NORMAL:
            self.current_item = self._render_as_points(positions, colors)
        
        elif mode == RenderMode.RINGS:
            self.current_item = self._render_as_rings(positions, colors, indices)
        
        elif mode == RenderMode.MESH:
            self.current_item = self._render_as_mesh(positions, colors, indices)
        
        if self.current_item:
            self.view.addItem(self.current_item)
        
        # 添加选中点高亮显示
        if self.editor:
            self._update_selection_highlight()
        
        # 更新统计信息
        self.update_stats(len(positions))
        
        # 更新方向指示器
        self._update_camera_params()
    
    def _update_selection_highlight(self):
        """更新选中点和锁定点高亮显示"""
        if not hasattr(self, 'sel_scatter'):
            self.sel_scatter = None
        if not hasattr(self, 'lock_scatter'):
            self.lock_scatter = None
        
        # === 更新锁定点高亮显示（橙色） ===
        if hasattr(self.gaussians, '_state_flags'):
            from gaussian_editor import LOCK_STATE, HIDE_STATE
            flags = self.gaussians._state_flags
            # 找到所有锁定且可见的点
            locked_mask = ((flags & LOCK_STATE) != 0) & ((flags & HIDE_STATE) == 0)
            locked_idx = torch.where(locked_mask)[0]
            
            if len(locked_idx) > 0:
                lpos = self.gaussians.get_xyz[locked_idx].detach().cpu().numpy()
                lpos = lpos - self.model_center  # 平移到原点
                lpos = self._apply_coord_transform_np(lpos)
                
                # 下采样（如果锁定太多点）
                if len(lpos) > 50000:
                    sample = np.random.choice(len(lpos), 50000, replace=False)
                    lpos = lpos[sample]
                
                if self.lock_scatter is None:
                    self.lock_scatter = gl.GLScatterPlotItem(
                        pos=lpos, 
                        color=(1.0, 0.5, 0.0, 1.0),  # 橙色表示锁定
                        size=5.0, 
                        pxMode=True
                    )
                    self.view.addItem(self.lock_scatter)
                    if self.lock_scatter not in self.current_items_list:
                        self.current_items_list.append(self.lock_scatter)
                else:
                    # 更新现有的lock_scatter数据
                    try:
                        self.lock_scatter.setData(pos=lpos, color=(1.0, 0.5, 0.0, 1.0), size=5.0)
                    except Exception:
                        # 如果更新失败，重新创建
                        self.view.removeItem(self.lock_scatter)
                        self.lock_scatter = gl.GLScatterPlotItem(
                            pos=lpos, 
                            color=(1.0, 0.5, 0.0, 1.0), 
                            size=5.0, 
                            pxMode=True
                        )
                        self.view.addItem(self.lock_scatter)
                        if self.lock_scatter not in self.current_items_list:
                            self.current_items_list.append(self.lock_scatter)
            else:
                # 没有锁定点时，移除lock_scatter
                if self.lock_scatter is not None:
                    try:
                        self.view.removeItem(self.lock_scatter)
                        if self.lock_scatter in self.current_items_list:
                            self.current_items_list.remove(self.lock_scatter)
                    except:
                        pass
                    self.lock_scatter = None
        
        # === 更新选中点高亮显示（黄色） ===
        sel_idx = self.editor.get_selected_indices()
        if len(sel_idx) > 0:
            # 过滤掉被隐藏的点 - 只显示可见且选中的点
            if hasattr(self.gaussians, '_state_flags'):
                from gaussian_editor import HIDE_STATE
                flags = self.gaussians._state_flags
                # 检查哪些选中的点没有被隐藏
                selected_flags = flags[sel_idx]
                visible_mask = (selected_flags & HIDE_STATE) == 0
                sel_idx = sel_idx[visible_mask]
            
            if len(sel_idx) == 0:
                # 所有选中的点都被隐藏了，移除高亮
                if self.sel_scatter is not None:
                    try:
                        self.view.removeItem(self.sel_scatter)
                        if self.sel_scatter in self.current_items_list:
                            self.current_items_list.remove(self.sel_scatter)
                    except:
                        pass
                    self.sel_scatter = None
                return
            
            spos = self.gaussians.get_xyz[sel_idx].detach().cpu().numpy()
            spos = spos - self.model_center  # 平移到原点
            spos = self._apply_coord_transform_np(spos)
            
            # 下采样（如果选中太多点）
            if len(spos) > 50000:
                sample = np.random.choice(len(spos), 50000, replace=False)
                spos = spos[sample]
            
            if self.sel_scatter is None:
                self.sel_scatter = gl.GLScatterPlotItem(
                    pos=spos, 
                    color=(1.0, 1.0, 0.0, 1.0), 
                    size=4.0, 
                    pxMode=True
                )
                self.view.addItem(self.sel_scatter)
                # 确保sel_scatter在列表中
                if self.sel_scatter not in self.current_items_list:
                    self.current_items_list.append(self.sel_scatter)
            else:
                # 更新现有的sel_scatter数据
                try:
                    self.sel_scatter.setData(pos=spos, color=(1.0, 1.0, 0.0, 1.0), size=4.0)
                except Exception as e:
                    # 如果更新失败，重新创建
                    self.view.removeItem(self.sel_scatter)
                    self.sel_scatter = gl.GLScatterPlotItem(
                        pos=spos, 
                        color=(1.0, 1.0, 0.0, 1.0), 
                        size=4.0, 
                        pxMode=True
                    )
                    self.view.addItem(self.sel_scatter)
                    if self.sel_scatter not in self.current_items_list:
                        self.current_items_list.append(self.sel_scatter)
        else:
            # 没有选中点时，移除sel_scatter
            if self.sel_scatter is not None:
                try:
                    self.view.removeItem(self.sel_scatter)
                    if self.sel_scatter in self.current_items_list:
                        self.current_items_list.remove(self.sel_scatter)
                except Exception:
                    pass  # 忽略移除错误
                self.sel_scatter = None
    
    def _render_as_points(self, positions, colors) -> gl.GLScatterPlotItem:
        """渲染为点云"""
        # 应用透明度
        if colors.shape[1] == 3:
            alpha_col = np.ones((len(colors), 1)) * self.settings.alpha
            colors = np.hstack([colors, alpha_col])
        else:
            colors[:, 3] = self.settings.alpha
        
        item = gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=self.settings.point_size,
            pxMode=True
        )
        return item
    
    def _render_as_ellipsoids(self, positions, colors, indices):
        """渲染为椭球体（使用椭球体渲染器）"""
        if self.ellipsoid_renderer is None:
            # 降级：使用更大的点来表示椭球体
            return self._render_as_ellipsoids_fallback(positions, colors, indices)
        
        try:
            # 获取高斯参数
            scales = self.gaussians.get_scaling[indices].detach().cpu().numpy()
            rotations = self.gaussians.get_rotation[indices].detach().cpu().numpy()
            
            # 应用椭球体缩放
            scales = scales * self.settings.ellipsoid_scale
            
            # 根据数量选择渲染方式
            if len(positions) < 100:
                # 少量：使用完整mesh
                items = self.ellipsoid_renderer.create_ellipsoid_meshes(
                    positions, rotations, scales, colors, max_ellipsoids=100
                )
                # 添加所有items
                for item in items:
                    self.view.addItem(item)
                    self.current_items_list.append(item)
                return None  # 返回None因为已经添加到视图
            
            elif len(positions) < 1000:
                # 中等：使用线框
                items = self.ellipsoid_renderer.create_ellipsoid_wireframes(
                    positions, rotations, scales, colors, max_ellipsoids=1000
                )
                for item in items:
                    self.view.addItem(item)
                    self.current_items_list.append(item)
                return None
            
            else:
                # 大量：使用批量渲染
                item = self.batched_ellipsoid_renderer.create_instanced_ellipsoids(
                    positions, rotations, scales, colors, max_instances=10000
                )
                return item
        
        except Exception as e:
            self.log(f"椭球体渲染失败，使用降级方案: {e}", "warning")
            return self._render_as_ellipsoids_fallback(positions, colors, indices)
    
    def _render_as_ellipsoids_fallback(self, positions, colors, indices):
        """椭球体渲染降级方案（使用大点）"""
        try:
            scales = self.gaussians.get_scaling[indices].detach().cpu().numpy()
            sizes = np.mean(scales, axis=1) * self.settings.ellipsoid_scale * 10.0
            sizes = np.clip(sizes, 1.0, 20.0)
        except Exception:
            sizes = self.settings.point_size * 2.0
        
        if colors.shape[1] == 3:
            alpha_col = np.ones((len(colors), 1)) * self.settings.alpha * 0.5
            colors = np.hstack([colors, alpha_col])
        
        item = gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=sizes if isinstance(sizes, np.ndarray) else self.settings.point_size * 2,
            pxMode=True
        )
        return item
    
    def _render_as_rings(self, positions, colors, indices) -> gl.GLScatterPlotItem:
        """渲染为环形（简化版）"""
        # 简化：使用空心点表示
        # 实际实现需要使用shader或者自定义mesh
        return self._render_as_points(positions, colors)
    
    def generate_mesh_from_points(self, positions, colors, method='poisson', depth=9, 
                                   scale=1.1, linear_fit=False, density_threshold=0.1):
        """从点云生成高质量mesh
        
        Args:
            positions: 点位置 [N, 3]
            colors: 点颜色 [N, 3] or [N, 4]
            method: 重建方法，'poisson' 或 'alpha_shape' 或 'ball_pivoting'
            depth: Poisson重建深度（越高越精细，推荐9-11）
            scale: Poisson缩放因子（控制外推程度）
            linear_fit: 是否使用线性拟合（更平滑但可能丢失细节）
            density_threshold: 密度阈值（0-1，用于过滤低密度区域）
        
        Returns:
            mesh_data: (vertices, faces, vertex_colors) 或 None
        """
        if not OPEN3D_AVAILABLE:
            self.log("Open3D未安装，无法生成mesh", "warning")
            return None
        
        try:
            self.log(f"正在使用{method}方法生成高质量mesh (depth={depth})...", "info")
            
            # 创建Open3D点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            
            # 设置颜色
            if colors.shape[1] == 4:
                colors = colors[:, :3]  # 去掉alpha通道
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 统计点云信息用于自适应参数
            points_np = np.asarray(pcd.points)
            distances = np.linalg.norm(points_np - points_np.mean(axis=0), axis=1)
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # 计算平均最近邻距离（用于自适应参数）
            pcd_tree_temp = o3d.geometry.KDTreeFlann(pcd)
            nn_distances = []
            sample_size = min(1000, len(positions))
            sample_indices = np.random.choice(len(positions), sample_size, replace=False)
            
            for idx in sample_indices:
                [k, idx_nn, dist_nn] = pcd_tree_temp.search_knn_vector_3d(points_np[idx], 10)
                if len(dist_nn) > 1:
                    nn_distances.append(np.mean(np.sqrt(dist_nn[1:])))
            
            avg_nn_dist = np.mean(nn_distances) if nn_distances else 0.1
            
            self.log(f"点云统计: 平均距离={avg_distance:.4f}, 平均邻域距离={avg_nn_dist:.4f}", "info")
            
            # 自适应估计法线（使用更优的参数）
            # 法线半径应该是平均邻域距离的2-3倍
            normal_radius = max(avg_nn_dist * 3.0, 0.05)
            normal_radius = min(normal_radius, avg_distance * 0.3)  # 不要太大
            
            self.log(f"使用法线估计半径: {normal_radius:.4f}", "info")
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius, 
                    max_nn=50  # 增加邻域点数以获得更准确的法线
                )
            )
            
            # 法线方向一致性调整（非常重要！）
            pcd.orient_normals_consistent_tangent_plane(k=15)
            
            # 根据方法生成mesh
            if method == 'poisson':
                # Poisson表面重建（高质量参数）
                self.log(f"执行Poisson重建 (depth={depth}, scale={scale})...", "info")
                
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, 
                    depth=depth,
                    width=0,  # 0表示自动计算
                    scale=scale,  # 控制外推
                    linear_fit=linear_fit  # 是否线性拟合
                )
                
                # 智能密度过滤
                if len(densities) > 0:
                    densities = np.asarray(densities)
                    
                    # 使用更智能的阈值（基于密度分布）
                    density_min = np.min(densities)
                    density_max = np.max(densities)
                    density_range = density_max - density_min
                    
                    # 自适应阈值：去除最低密度区域
                    adaptive_threshold = density_min + density_range * density_threshold
                    
                    vertices_to_remove = densities < adaptive_threshold
                    removed_count = np.sum(vertices_to_remove)
                    
                    self.log(f"密度过滤: 移除 {removed_count}/{len(densities)} 个低密度顶点 (阈值={adaptive_threshold:.4f})", "info")
                    
                    mesh.remove_vertices_by_mask(vertices_to_remove)
                
            elif method == 'alpha_shape':
                # Alpha Shape（自适应alpha）
                alpha = avg_nn_dist * 2.0  # 自适应alpha值
                self.log(f"执行Alpha Shape重建 (alpha={alpha:.4f})...", "info")
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha
                )
                
            elif method == 'ball_pivoting':
                # Ball Pivoting算法（自适应半径）
                base_radius = avg_nn_dist * 1.5
                radii = [base_radius * r for r in [0.5, 1.0, 2.0, 4.0]]
                
                self.log(f"执行Ball Pivoting重建 (半径={radii})...", "info")
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd,
                    o3d.utility.DoubleVector(radii)
                )
                
            else:
                self.log(f"未知的mesh生成方法: {method}", "error")
                return None
            
            # Mesh后处理（提高质量）
            self.log("执行Mesh后处理优化...", "info")
            
            # 1. 移除重复顶点
            mesh.remove_duplicated_vertices()
            
            # 2. 移除退化三角形
            mesh.remove_degenerate_triangles()
            
            # 3. 移除不相连的小块（孤岛）
            triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            
            # 只保留最大的几个连通分量
            largest_clusters = np.argsort(cluster_n_triangles)[::-1]
            min_cluster_size = max(100, len(mesh.triangles) * 0.01)  # 至少保留1%的三角形
            
            triangles_to_keep = []
            for cluster_idx in largest_clusters:
                if cluster_n_triangles[cluster_idx] >= min_cluster_size:
                    triangles_to_keep.append(triangle_clusters == cluster_idx)
                    
            if triangles_to_keep:
                triangles_mask = np.logical_or.reduce(triangles_to_keep)
                mesh.remove_triangles_by_mask(~triangles_mask)
                self.log(f"保留了 {len(triangles_to_keep)} 个主要连通分量", "info")
            
            # 4. 平滑处理（可选，轻微平滑以去除噪声）
            mesh = mesh.filter_smooth_simple(number_of_iterations=2)
            
            # 5. 重新计算法线
            mesh.compute_vertex_normals()
            
            # 如果mesh没有颜色，从最近的点插值颜色（使用多点平均）
            if not mesh.has_vertex_colors():
                self.log("为mesh顶点插值颜色...", "info")
                vertices = np.asarray(mesh.vertices)
                pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                vertex_colors = np.zeros((len(vertices), 3))
                
                # 使用多个最近邻的平均颜色（更平滑）
                k_neighbors = 5
                for i, v in enumerate(vertices):
                    [k, idx, dist] = pcd_tree.search_knn_vector_3d(v, k_neighbors)
                    if len(idx) > 0:
                        # 基于距离的加权平均
                        weights = 1.0 / (np.array(dist) + 1e-6)
                        weights = weights / weights.sum()
                        vertex_colors[i] = np.average(colors[idx], axis=0, weights=weights)
                
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            # 提取mesh数据
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vertex_colors = np.asarray(mesh.vertex_colors)
            
            self.log(f"✓ 高质量Mesh生成成功: {len(vertices)}个顶点, {len(faces)}个面", "info")
            
            return (vertices, faces, vertex_colors)
            
        except Exception as e:
            import traceback
            self.log(f"Mesh生成失败: {e}\n{traceback.format_exc()}", "error")
            return None
    
    def _render_as_mesh(self, positions, colors, indices):
        """渲染为mesh（仅渲染已生成的mesh，不自动生成）"""
        if not PG_AVAILABLE:
            self.log("pyqtgraph不可用", "error")
            return None
        
        # 检查是否有缓存的mesh
        if self.cached_mesh is None:
            self.log("没有可用的mesh，请先通过菜单生成mesh (Ctrl+M)", "warning")
            # 降级为点云显示
            return self._render_as_points(positions, colors)
        
        vertices, faces, vertex_colors = self.cached_mesh
        if len(vertices) == 0 or len(faces) == 0:
            self.log("Mesh数据无效，请重新生成", "warning")
            return self._render_as_points(positions, colors)
        
        self.log(f"渲染Mesh: {len(vertices)}个顶点, {len(faces)}个面", "info")
        
        # 创建pyqtgraph mesh item
        try:
            # 准备mesh数据
            mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
            
            # 设置顶点颜色
            if vertex_colors is not None and len(vertex_colors) == len(vertices):
                # 添加alpha通道
                if vertex_colors.shape[1] == 3:
                    alpha_col = np.ones((len(vertex_colors), 1)) * self.settings.alpha
                    vertex_colors_rgba = np.hstack([vertex_colors, alpha_col])
                else:
                    vertex_colors_rgba = vertex_colors.copy()
                    vertex_colors_rgba[:, 3] = self.settings.alpha
                
                mesh_data.setVertexColors(vertex_colors_rgba)
            
            # 创建mesh item
            mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,  # 平滑着色
                shader='shaded',  # 使用带光照的shader
                glOptions='opaque'  # 不透明
            )
            
            self.log(f"Mesh渲染成功: {len(vertices)}个顶点, {len(faces)}个面", "info")
            return mesh_item
            
        except Exception as e:
            import traceback
            self.log(f"Mesh渲染失败: {e}\n{traceback.format_exc()}", "error")
            # 降级为点云显示
            return self._render_as_points(positions, colors)
    
    def enable_mesh_mode(self):
        """启用mesh模式（在mesh生成成功后调用）"""
        if hasattr(self, 'btn_mesh'):
            self.btn_mesh.setEnabled(True)
            self.btn_mesh.setToolTip("切换到Mesh渲染模式")
        self.log("Mesh模式已启用", "info")
    
    def disable_mesh_mode(self):
        """禁用mesh模式（在清除缓存后调用）"""
        if hasattr(self, 'btn_mesh'):
            self.btn_mesh.setEnabled(False)
            self.btn_mesh.setToolTip("请先通过菜单生成mesh (Ctrl+M)")
        # 如果当前是mesh模式，切换回点云模式
        if self.settings.render_mode == RenderMode.MESH:
            self.set_render_mode(RenderMode.POINT_CLOUD)
    
    def clear_mesh_cache(self):
        """清除mesh缓存"""
        self.cached_mesh = None
        self.disable_mesh_mode()
        self.log("Mesh缓存已清除", "info")
    
    def export_mesh(self, filepath, format='obj'):
        """导出mesh到文件
        
        Args:
            filepath: 输出文件路径
            format: 文件格式 ('obj', 'ply', 'stl')
        """
        if not OPEN3D_AVAILABLE:
            self.log("Open3D未安装，无法导出mesh", "error")
            return False
        
        if self.cached_mesh is None:
            self.log("没有可导出的mesh，请先切换到mesh模式", "warning")
            return False
        
        try:
            vertices, faces, vertex_colors = self.cached_mesh
            
            # 创建Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            if vertex_colors is not None and len(vertex_colors) > 0:
                if vertex_colors.shape[1] == 4:
                    vertex_colors = vertex_colors[:, :3]  # 去掉alpha
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            # 导出
            success = o3d.io.write_triangle_mesh(filepath, mesh)
            
            if success:
                self.log(f"Mesh已导出到: {filepath}", "info")
            else:
                self.log(f"Mesh导出失败", "error")
            
            return success
            
        except Exception as e:
            import traceback
            self.log(f"Mesh导出失败: {e}\n{traceback.format_exc()}", "error")
            return False
    
    def update_stats(self, num_points: int):
        """更新统计信息"""
        mode_name = self.settings.render_mode.name
        stats_text = f"模式: {mode_name} | 点数: {num_points:,} | LOD: {'开' if self.settings.lod_enabled else '关'}"
        self.stats_label.setText(stats_text)
    
    def log(self, message: str, level: str = "info"):
        """日志输出"""
        if hasattr(self.parent(), 'log'):
            self.parent().log(message, level)
        else:
            print(f"[{level.upper()}] {message}")
    
    def _update_camera_params(self):
        """更新相机参数（从pyqtgraph GL视图获取）"""
        try:
            view_matrix = self.view.viewMatrix()
            proj_matrix = self.view.projectionMatrix()
            
            # 转换为numpy数组（pyqtgraph使用行主序，需要转置）
            view_np = np.array(view_matrix.data()).reshape(4, 4).T
            proj_np = np.array(proj_matrix.data()).reshape(4, 4).T
            
            self.set_camera_params(view_np, proj_np)
            
            # 更新方向指示器
            if hasattr(self, 'orientation_indicator'):
                self.orientation_indicator.set_rotation(view_np)
            # 在SPLAT模式下，交互时重渲
            if self.settings.render_mode == RenderMode.SPLAT:
                self._render_as_splat_image()
        except Exception as e:
            # 如果获取失败，静默处理
            pass
    
    # ==================== 选择功能（完整实现） ====================
    
    def set_selection_mode(self, enabled: bool, selection_type: str = 'box'):
        """设置选择模式"""
        self.selection_mode = enabled
        # 如果selection_type为None，使用默认值'box'
        if selection_type is None:
            self.selection_type = 'box'
        else:
            self.selection_type = selection_type.lower()
        
        # 如果禁用选择模式，清除当前选择状态
        if not enabled:
            self.is_selecting = False
            self.select_start = None
            self.select_end = None
            self.select_path = []
            self.view.update()
        
        # 更新相机参数
        self._update_camera_params()
    
    def on_mouse_press(self, evt):
        """鼠标按下"""
        from PyQt5.QtCore import Qt
        
        # 更新相机参数（在每次交互前更新）
        self._update_camera_params()
        
        if evt.button() == Qt.LeftButton and self.selection_mode:
            self.is_selecting = True
            # 转换为视图坐标
            local_pos = self.view.mapFromGlobal(evt.globalPos())
            self.select_start = local_pos
            self.select_end = local_pos
            self.select_path = [local_pos]
            # 更新覆盖层显示选择框
            if hasattr(self, 'overlay'):
                self.overlay.set_selection(self.select_start, self.select_end, self.selection_type, self.select_path)
        else:
            # 调用原始处理
            if self._original_mouse_press:
                self._original_mouse_press(evt)
    
    def on_mouse_move(self, evt):
        """鼠标移动"""
        if self.is_selecting and self.selection_mode:
            # 更新相机参数（在拖动过程中实时更新）
            self._update_camera_params()
            
            local_pos = self.view.mapFromGlobal(evt.globalPos())
            self.select_end = local_pos
            if self.selection_type in ['lasso', 'brush']:
                self.select_path.append(local_pos)
                # 限制路径点数量，避免内存过大
                if len(self.select_path) > 1000:
                    # 每10个点保留1个
                    self.select_path = self.select_path[::10] + [self.select_path[-1]]
            # 更新覆盖层显示选择框
            if hasattr(self, 'overlay'):
                self.overlay.set_selection(self.select_start, self.select_end, self.selection_type, self.select_path)
        else:
            # 调用原始处理
            if self._original_mouse_move:
                self._original_mouse_move(evt)
    
    def on_mouse_release(self, evt):
        """鼠标释放"""
        from PyQt5.QtCore import Qt
        
        if evt.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            
            # 更新相机参数（在释放前最后更新一次）
            self._update_camera_params()
            
            if self.editor and self.gaussians and self.camera_view is not None and self.camera_proj is not None:
                visible_indices = self.editor.get_visible_indices()
                if len(visible_indices) > 0:
                    # 使用GPU tensor直接处理（避免CPU转换），并应用中心化与坐标变换
                    points_3d_tensor = self.gaussians.get_xyz[visible_indices].detach()
                    # 平移到原点（保持设备与精度）
                    center_tensor = torch.from_numpy(self.model_center).to(device=points_3d_tensor.device, dtype=points_3d_tensor.dtype)
                    points_3d_tensor = points_3d_tensor - center_tensor
                    points_3d_tensor = self._apply_coord_transform_tensor(points_3d_tensor)
                    
                    # 对于大规模点云，先下采样以提高速度
                    max_points = 500000  # 最多处理50万点
                    if len(visible_indices) > max_points:
                        # 随机下采样
                        sample_indices = torch.randperm(len(visible_indices), device=visible_indices.device)[:max_points]
                        sampled_visible = visible_indices[sample_indices]
                        points_3d_tensor = self.gaussians.get_xyz[sampled_visible].detach()
                        # 平移到原点（保持设备与精度）
                        center_tensor = torch.from_numpy(self.model_center).to(device=points_3d_tensor.device, dtype=points_3d_tensor.dtype)
                        points_3d_tensor = points_3d_tensor - center_tensor
                        points_3d_tensor = self._apply_coord_transform_tensor(points_3d_tensor)
                        use_sampled = True
                    else:
                        sampled_visible = visible_indices
                        use_sampled = False
                    
                    # 投影（GPU加速）
                    if self.selection_type == 'box':
                        screen_rect = self._get_screen_rect(self.select_start, self.select_end)
                        # 检查矩形是否有效（至少有一些面积）
                        if abs(screen_rect[2] - screen_rect[0]) > 2 and abs(screen_rect[3] - screen_rect[1]) > 2:
                            mask = self._points_in_box_gpu(points_3d_tensor, screen_rect)
                        else:
                            mask = np.zeros(len(points_3d_tensor), dtype=bool)
                    elif self.selection_type == 'lasso':
                        # 套索需要CPU处理（多边形测试）
                        if len(self.select_path) >= 3:
                            points_3d_cpu = points_3d_tensor.detach().cpu().numpy()
                            mask = self._points_in_lasso(points_3d_cpu, self.select_path)
                        else:
                            mask = np.zeros(len(points_3d_tensor), dtype=bool)
                    elif self.selection_type == 'brush':
                        if len(self.select_path) >= 1:
                            mask = self._points_in_brush_gpu(points_3d_tensor, self.select_path)
                        else:
                            mask = np.zeros(len(points_3d_tensor), dtype=bool)
                    else:
                        mask = np.zeros(len(points_3d_tensor), dtype=bool)
                    
                    # 应用选择
                    if use_sampled:
                        selected_abs_indices = sampled_visible[mask]
                    else:
                        selected_abs_indices = visible_indices[mask]
                    
                    if len(selected_abs_indices) > 0:
                        # 创建选择操作
                        device = self.gaussians.get_xyz.device
                        num_points = self.gaussians.get_xyz.shape[0]
                        mask_tensor = torch.zeros(num_points, dtype=torch.bool, device=device)
                        # 确保索引在有效范围内
                        valid_mask = selected_abs_indices < num_points
                        if valid_mask.any():
                            mask_tensor[selected_abs_indices[valid_mask]] = True
                            
                            from gaussian_editor import EditSelectOpType, SplatSelectionOp, UndoRedoSystem
                            op = SplatSelectionOp(self.gaussians, EditSelectOpType.ADD, precomputed_mask=mask_tensor)
                            UndoRedoSystem().add(op)
                            self.update_view()
                            self.selection_changed.emit()
                            self.log(f"已选择 {len(selected_abs_indices[valid_mask])} 个点")
                        else:
                            self.log("未选择任何点（索引超出范围）", "warning")
                    else:
                        self.log("未选择任何点", "info")
            
            self.select_start = None
            self.select_end = None
            self.select_path = []
            # 清除覆盖层的选择框
            if hasattr(self, 'overlay'):
                self.overlay.clear_selection()
        else:
            # 调用原始处理
            if self._original_mouse_release:
                self._original_mouse_release(evt)
    
    def _get_screen_rect(self, start, end):
        """获取屏幕矩形区域"""
        x1, y1 = min(start.x(), end.x()), min(start.y(), end.y())
        x2, y2 = max(start.x(), end.x()), max(start.y(), end.y())
        return (x1, y1, x2, y2)
    
    def _get_screen_bounds(self):
        """获取屏幕边界"""
        rect = self.view.geometry()
        return rect.width(), rect.height()
    
    def _project_3d_to_2d(self, points_3d):
        """将3D点投影到2D屏幕空间（GPU加速版本）"""
        if self.camera_view is None or self.camera_proj is None:
            return None, None
        
        # 如果points_3d是torch tensor，使用GPU加速
        if isinstance(points_3d, torch.Tensor):
            device = points_3d.device
            # 确保输入tensor不需要梯度
            if points_3d.requires_grad:
                points_3d = points_3d.detach()
            
            view_proj = torch.from_numpy(self.camera_proj).float().to(device) @ \
                       torch.from_numpy(self.camera_view).float().to(device)
            points_4d = torch.cat([points_3d, torch.ones(len(points_3d), 1, device=device)], dim=1)
            projected = (view_proj @ points_4d.T).T
            w = torch.clamp(projected[:, 3], min=1e-6)
            ndc = projected[:, :3] / w.unsqueeze(1)
            screen = (ndc[:, :2] + 1.0) * 0.5
            screen[:, 1] = 1.0 - screen[:, 1]
            return screen.detach().cpu().numpy(), w.detach().cpu().numpy()
        else:
            # CPU版本（向量化）
            view_proj = self.camera_proj @ self.camera_view
            points_4d = np.column_stack([points_3d, np.ones(len(points_3d))])
            projected = (view_proj @ points_4d.T).T
            w = np.maximum(projected[:, 3], 1e-6)
            ndc = projected[:, :3] / w[:, np.newaxis]
            screen = (ndc[:, :2] + 1.0) * 0.5
            screen[:, 1] = 1.0 - screen[:, 1]
            return screen, w
    
    def _points_in_box_gpu(self, points_3d_tensor, screen_rect):
        """GPU加速的框选"""
        screen_coords, depths = self._project_3d_to_2d(points_3d_tensor)
        if screen_coords is None:
            return np.zeros(len(points_3d_tensor), dtype=bool)
        
        w, h = self._get_screen_bounds()
        x1, y1, x2, y2 = screen_rect
        
        sx1, sy1 = x1 / w, y1 / h
        sx2, sy2 = x2 / w, y2 / h
        
        mask = (screen_coords[:, 0] >= sx1) & (screen_coords[:, 0] <= sx2) & \
               (screen_coords[:, 1] >= sy1) & (screen_coords[:, 1] <= sy2)
        
        return mask
    
    def _points_in_lasso(self, points_3d, path):
        """检查3D点是否在套索路径内（向量化射线法）"""
        screen_coords, depths = self._project_3d_to_2d(points_3d)
        if screen_coords is None or len(path) < 3:
            return np.zeros(len(points_3d), dtype=bool)
        
        w, h = self._get_screen_bounds()
        path_norm = np.array([(p.x() / w, p.y() / h) for p in path])
        
        # 向量化射线法（使用numpy广播）
        px = screen_coords[:, 0:1]  # [N, 1]
        py = screen_coords[:, 1:2]  # [N, 1]
        
        # 多边形的边
        path_xy = path_norm[:, :2]  # [M, 2]
        path_xy_next = np.roll(path_xy, -1, axis=0)  # [M, 2]
        
        x1, y1 = path_xy[:, 0], path_xy[:, 1]  # [M]
        x2, y2 = path_xy_next[:, 0], path_xy_next[:, 1]  # [M]
        
        # 向量化计算
        py_expanded = py  # [N, 1]
        y1_expanded = y1[np.newaxis, :]  # [1, M]
        y2_expanded = y2[np.newaxis, :]  # [1, M]
        
        # 计算边的dy和dx
        dy = (y2 - y1)[np.newaxis, :]  # [1, M]
        dx = (x2 - x1)[np.newaxis, :]  # [1, M]
        
        # 排除水平边（dy接近0的情况），避免错误的交点计算
        # 使用一个小的阈值来判断是否为水平边
        is_horizontal = np.abs(dy) < 1e-8  # [1, M]
        
        # 检查射线是否穿过边（排除水平边）
        y1_gt_py = (y1_expanded > py_expanded)  # [N, M]
        y2_gt_py = (y2_expanded > py_expanded)  # [N, M]
        crosses_y = (y1_gt_py != y2_gt_py) & (~is_horizontal)  # [N, M]，排除水平边
        
        # 计算交点的x坐标（只对非水平边计算）
        # 对于非水平边，使用正常的交点计算
        # 使用 np.errstate 来抑制除以零警告，并使用安全的除法
        with np.errstate(divide='ignore', invalid='ignore'):
            # 确保 dy 不为零（添加小的 epsilon 来避免除以零）
            dy_safe = np.where(np.abs(dy) > 1e-10, dy, 1.0)
            ratio = np.where(~is_horizontal, 
                            (py_expanded - y1_expanded) / dy_safe, 
                            0.0)  # [N, M]
        x1_expanded = x1[np.newaxis, :]  # [1, M]
        intersect_x = x1_expanded + dx * ratio  # [N, M]
        
        # 检查交点是否在点的右侧（排除水平边）
        px_expanded = px  # [N, 1]
        intersects_right = (intersect_x > px_expanded) & crosses_y  # [N, M]
        
        # 统计每个点的交点数量
        counts = np.sum(intersects_right, axis=1)  # [N]
        mask = (counts % 2) == 1
        
        return mask
    
    def _points_in_brush_gpu(self, points_3d_tensor, path, radius=0.02):
        """GPU加速的笔刷选择"""
        screen_coords, depths = self._project_3d_to_2d(points_3d_tensor)
        if screen_coords is None or len(path) < 1:
            return np.zeros(len(points_3d_tensor), dtype=bool)
        
        w, h = self._get_screen_bounds()
        path_norm = np.array([(p.x() / w, p.y() / h) for p in path])
        
        # 向量化距离计算
        diff = screen_coords[:, np.newaxis, :] - path_norm[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        min_dists = np.min(dists_sq, axis=1)
        
        mask = np.sqrt(min_dists) < radius
        return mask
    
    def resizeEvent(self, event):
        """窗口大小改变时更新覆盖层和方向指示器"""
        super().resizeEvent(event)
        if hasattr(self, 'overlay') and hasattr(self, 'view_container'):
            # 确保覆盖层与容器对齐
            self.overlay.setGeometry(0, 0, self.view_container.width(), self.view_container.height())
            self.overlay.raise_()  # 确保在最上层
        
        # 重新定位方向指示器到右上角
        if hasattr(self, 'orientation_indicator'):
            self._position_orientation_indicator()
            self.orientation_indicator.raise_()  # 确保在最上层
        
        # 调整SPLAT覆盖层
        if hasattr(self, 'splat_label') and hasattr(self, 'view_container'):
            self.splat_label.setGeometry(0, 0, self.view_container.width(), self.view_container.height())
            try:
                self.view.stackUnder(self.splat_label)
                if hasattr(self, 'overlay'):
                    self.splat_label.stackUnder(self.overlay)
            except Exception:
                pass
            if self.settings.render_mode == RenderMode.SPLAT:
                self._render_as_splat_image()
    
    def set_camera_params(self, view_matrix=None, proj_matrix=None):
        """设置相机参数（兼容接口）"""
        self.camera_view = view_matrix
        self.camera_proj = proj_matrix

    def _apply_coord_transform_np(self, positions: np.ndarray) -> np.ndarray:
        """对numpy坐标应用当前坐标变换"""
        try:
            if positions is None or len(positions) == 0:
                return positions
            return positions @ self._coord_matrix.T
        except Exception:
            return positions

    def _apply_coord_transform_tensor(self, points: torch.Tensor) -> torch.Tensor:
        """对torch坐标应用当前坐标变换（保持设备/精度）"""
        try:
            if points is None or points.numel() == 0:
                return points
            device = points.device
            dtype = points.dtype
            M = torch.from_numpy(self._coord_matrix).to(device=device, dtype=dtype)
            return points @ M.T
        except Exception:
            return points

    def _compute_world_view_for_3dgs(self) -> tuple:
        """构造与点云显示一致的3DGS world_view 与 projection。
        点云显示使用 display = (world - center) @ M^T，且GL视图矩阵作用在display坐标上。
        因此令 T = [[M^T, -center @ M^T],[0,0,0,1]]，world_view = V_display @ T，projection=P_display。
        返回 (world_view_np, proj_np)
        """
        # 读取pyqtgraph当前矩阵（列主 -> 转置成行主）
        Vd = np.array(self.view.viewMatrix().data()).reshape(4, 4).T
        P = np.array(self.view.projectionMatrix().data()).reshape(4, 4).T
        # 构造T
        M = self._coord_matrix.astype(np.float32)  # 3x3
        Mt = M.T
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Mt
        T[:3, 3] = -self.model_center.astype(np.float32) @ Mt
        # world坐标视图矩阵
        Vw = Vd @ T
        return Vw, P

    def _render_as_splat_image(self):
        """使用3DGS渲染器渲染为图像并覆盖显示（严格与点云坐标对齐）。"""
        try:
            if self.gaussians is None:
                return
            if not torch.cuda.is_available():
                if hasattr(self, 'splat_label') and self.splat_label.isVisible():
                    self.splat_label.hide()
                return

            w = max(1, int(self.view_container.width()))
            h = max(1, int(self.view_container.height()))

            Vw_np, P_np = self._compute_world_view_for_3dgs()

            # 从投影矩阵近似反推FoV，若失败则用视图的fov
            try:
                px = float(P_np[0, 0]); py = float(P_np[1, 1])
                tan_half_fovx = 1.0 / px if abs(px) > 1e-6 else None
                tan_half_fovy = 1.0 / py if abs(py) > 1e-6 else None
                if tan_half_fovx is None or tan_half_fovy is None:
                    fov_deg = float(self.view.opts.get('fov', 60))
                    fovy = np.deg2rad(fov_deg)
                    aspect = w / max(1.0, float(h))
                    fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)
                else:
                    fovx = 2.0 * np.arctan(tan_half_fovx)
                    fovy = 2.0 * np.arctan(tan_half_fovy)
            except Exception:
                fov_deg = float(self.view.opts.get('fov', 60))
                fovy = np.deg2rad(fov_deg)
                aspect = w / max(1.0, float(h))
                fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)

            world_view = torch.tensor(Vw_np, dtype=torch.float32, device='cuda')
            proj = torch.tensor(P_np, dtype=torch.float32, device='cuda')
            full_proj = world_view @ proj

            from gaussiansplatting.scene.cameras import MiniCam
            cam = MiniCam(w, h, fovy, fovx, 0.01, 100.0, world_view, full_proj)

            from gaussiansplatting.gaussian_renderer import render as gs_render
            pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
            bg = torch.zeros(3, device='cuda')
            out = gs_render(cam, self.gaussians, pipe, bg, scaling_modifier=self.settings.splat_scaling_modifier)
            img = out["render"].detach().clamp(0.0, 1.0).permute(1, 2, 0).contiguous().cpu().numpy()
            img_u8 = (img * 255.0).astype(np.uint8)

            from PyQt5.QtGui import QImage, QPixmap
            qimg = QImage(img_u8.data, img_u8.shape[1], img_u8.shape[0], int(img_u8.shape[1] * 3), QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())

            self.splat_label.setGeometry(0, 0, self.view_container.width(), self.view_container.height())
            self.splat_label.setPixmap(pix)
            self.splat_label.show()

            try:
                self.view.stackUnder(self.splat_label)
                if hasattr(self, 'overlay'):
                    self.splat_label.stackUnder(self.overlay)
            except Exception:
                pass

        except Exception:
            try:
                self.splat_label.hide()
            except Exception:
                pass

