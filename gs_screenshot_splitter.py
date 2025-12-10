#!/usr/bin/env python3
"""
3DGS模型截图分割功能
实现视角截图、射线检测、XZ平面约束等功能
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QDoubleSpinBox, QFrame, QSizePolicy, QFileDialog, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QPolygon
from PIL import Image
import json
import os
import base64
import io
import requests
import cv2


class ScreenshotPreviewWidget(QWidget):
    """截图预览面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 增大预览面板尺寸
        self.setMinimumSize(600, 500)
        self.setMaximumSize(800, 700)
        # 简化样式
        self.setStyleSheet("background-color: #2B2B2B; border: 1px solid #555;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 标题栏（简化）
        title_layout = QHBoxLayout()
        title = QLabel("截图预览")
        title.setStyleSheet("color: #FFF; font-weight: bold; font-size: 14px;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # 预览图像标签（增大尺寸）
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1E1E1E; border: 1px solid #444;")
        self.preview_label.setMinimumSize(580, 350)
        self.preview_label.setText("等待截图...")
        self.preview_label.setStyleSheet(
            "background-color: #1E1E1E; border: 1px solid #444; color: #999; font-size: 12px;"
        )
        layout.addWidget(self.preview_label)
        
        # 信息显示（简化）
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #AAA; font-size: 11px; padding: 5px;")
        self.info_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.info_label)
        
        # 分割相关控件（简化布局）
        seg_group = QGroupBox("分割设置")
        seg_group.setStyleSheet("QGroupBox { color: #AAA; font-size: 12px; border: 1px solid #444; padding-top: 10px; }")
        seg_layout = QVBoxLayout()
        seg_layout.setSpacing(8)
        
        # 分割提示词输入（简化）
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel("提示词:")
        prompt_label.setStyleSheet("color: #AAA; font-size: 11px; min-width: 60px;")
        self.prompt_input = QLineEdit("building")
        self.prompt_input.setPlaceholderText("如: building, car, tree")
        self.prompt_input.setStyleSheet(
            "background-color: #1E1E1E; color: #FFF; border: 1px solid #444; padding: 5px; font-size: 11px;"
        )
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_input)
        seg_layout.addLayout(prompt_layout)
        
        # 分割按钮和状态（水平布局）
        btn_status_layout = QHBoxLayout()
        self.segment_btn = QPushButton("分割对象")
        self.segment_btn.setStyleSheet(
            "background-color: #6C5CE7; color: white; padding: 8px 15px; border: none; font-size: 11px; font-weight: bold;"
        )
        self.segment_btn.clicked.connect(self.on_segment_clicked)
        btn_status_layout.addWidget(self.segment_btn)
        
        self.segment_status_label = QLabel("")
        self.segment_status_label.setStyleSheet("color: #AAA; font-size: 11px; padding-left: 10px;")
        self.segment_status_label.setAlignment(Qt.AlignLeft)
        btn_status_layout.addWidget(self.segment_status_label)
        btn_status_layout.addStretch()
        seg_layout.addLayout(btn_status_layout)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        # 按钮布局（简化）
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.download_btn = QPushButton("下载图片")
        self.download_btn.setStyleSheet(
            "background-color: #0984E3; color: white; padding: 8px 20px; border: none; font-size: 11px;"
        )
        self.close_btn = QPushButton("关闭")
        self.close_btn.setStyleSheet(
            "background-color: #D63031; color: white; padding: 8px 20px; border: none; font-size: 11px;"
        )
        btn_layout.addStretch()
        btn_layout.addWidget(self.download_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
        
        self.current_image = None
        self.corner_points = []  # 四个角点（像素坐标）
        self.segmentation_mask = None  # 分割mask（numpy数组）
        self.segmentation_result = None  # 分割结果（GeoJSON）
        self.sam3_api_url = "http://localhost:5000"  # SAM3 API地址
        
    def set_preview_image(self, image: QImage, corner_points: List[QPoint] = None, show_segmentation: bool = True):
        """
        设置预览图像
        
        Args:
            image: 预览图像
            corner_points: 四个角点的像素坐标
            show_segmentation: 是否显示分割结果
        """
        self.current_image = image
        self.corner_points = corner_points if corner_points else []
        
        # 创建带标记的图像
        if image:
            pixmap = QPixmap.fromImage(image)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 如果存在分割mask，绘制分割结果
            if show_segmentation and self.segmentation_mask is not None:
                self._draw_segmentation_mask(painter, pixmap.width(), pixmap.height())
            
            # 绘制四个角点标记
            if self.corner_points and len(self.corner_points) == 4:
                # 绘制红色角点标记（四分之一圆）
                pen = QPen(QColor(255, 0, 0), 3)
                painter.setPen(pen)
                brush = QBrush(QColor(255, 0, 0, 100))
                painter.setBrush(brush)
                
                radius = 15
                for i, point in enumerate(self.corner_points):
                    # 绘制四分之一圆在四个角
                    if i == 0:  # 左上角
                        painter.drawPie(point.x() - radius, point.y() - radius, 
                                       radius * 2, radius * 2, 90 * 16, 90 * 16)
                    elif i == 1:  # 右上角
                        painter.drawPie(point.x() - radius, point.y() - radius, 
                                       radius * 2, radius * 2, 0, 90 * 16)
                    elif i == 2:  # 右下角
                        painter.drawPie(point.x() - radius, point.y() - radius, 
                                       radius * 2, radius * 2, 270 * 16, 90 * 16)
                    elif i == 3:  # 左下角
                        painter.drawPie(point.x() - radius, point.y() - radius, 
                                       radius * 2, radius * 2, 180 * 16, 90 * 16)
            
            painter.end()
            
            # 缩放以适应标签大小
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
    
    def _draw_segmentation_mask(self, painter: QPainter, image_width: int, image_height: int):
        """在图像上绘制分割mask"""
        if self.segmentation_mask is None:
            return
        
        try:
            # 将mask缩放到图像尺寸
            mask_resized = cv2.resize(
                self.segmentation_mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 为每个检测到的对象使用不同颜色
            unique_labels = np.unique(mask_resized)
            colors = [
                QColor(255, 0, 0, 150),    # 红色
                QColor(0, 255, 0, 150),    # 绿色
                QColor(0, 0, 255, 150),    # 蓝色
                QColor(255, 255, 0, 150),  # 黄色
                QColor(255, 0, 255, 150),  # 洋红
                QColor(0, 255, 255, 150),  # 青色
            ]
            
            for idx, label in enumerate(unique_labels):
                if label == 0:  # 背景
                    continue
                
                color = colors[idx % len(colors)]
                brush = QBrush(color)
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)
                
                # 找到该标签的轮廓
                mask_label = (mask_resized == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 绘制每个轮廓
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    # 转换为QPolygon
                    points = [QPoint(int(p[0][0]), int(p[0][1])) for p in contour]
                    polygon = QPolygon(points)
                    painter.drawPolygon(polygon)
        except Exception as e:
            print(f"绘制分割mask失败: {e}")
    
    def on_segment_clicked(self):
        """处理分割按钮点击"""
        if self.current_image is None:
            self.segment_status_label.setText("错误: 没有可用的图像")
            self.segment_status_label.setStyleSheet("color: #FF6B6B; font-size: 10px;")
            return
        
        prompt = self.prompt_input.text().strip()
        if not prompt:
            prompt = "building"  # 默认提示词
        
        self.segment_btn.setEnabled(False)
        self.segment_status_label.setText("正在分割...")
        self.segment_status_label.setStyleSheet("color: #5BA3D8; font-size: 10px;")
        
        # 在后台线程中执行分割
        from PyQt5.QtCore import QThread, pyqtSignal
        
        class SegmentThread(QThread):
            finished = pyqtSignal(dict)
            
            def __init__(self, image, prompt, api_url):
                super().__init__()
                self.image = image
                self.prompt = prompt
                self.api_url = api_url
            
            def run(self):
                result = call_sam3_api(self.image, self.prompt, self.api_url)
                self.finished.emit(result)
        
        self.segment_thread = SegmentThread(self.current_image, prompt, self.sam3_api_url)
        self.segment_thread.finished.connect(self.on_segmentation_finished)
        self.segment_thread.start()
    
    def on_segmentation_finished(self, result: dict):
        """处理分割完成"""
        self.segment_btn.setEnabled(True)
        
        if result.get('success', False):
            # 解析分割结果
            geojson = result.get('geojson', {})
            summary = result.get('summary', '')
            
            # 从GeoJSON创建mask
            if geojson and 'features' in geojson:
                self.segmentation_result = geojson
                self.segmentation_mask = self._geojson_to_mask(geojson, 
                                                               self.current_image.width(),
                                                               self.current_image.height())
                
                num_objects = len(geojson.get('features', []))
                self.segment_status_label.setText(f"√ {summary} 检测到{num_objects}个对象轮廓")
                self.segment_status_label.setStyleSheet("color: #51CF66; font-size: 10px;")
                
                # 更新预览图像以显示分割结果
                self.set_preview_image(self.current_image, self.corner_points, show_segmentation=True)
                
                # 通知splitter分割完成（如果splitter有cesium_widget，可以计算3D轮廓）
                if hasattr(self, '_splitter_ref'):
                    splitter = self._splitter_ref()
                    if splitter and hasattr(splitter, 'on_segmentation_mask_ready'):
                        print(f"[ScreenshotPreviewWidget] 通知splitter分割完成，mask尺寸: {self.segmentation_mask.shape if self.segmentation_mask is not None else 'None'}")
                        splitter.on_segmentation_mask_ready(
                            self.segmentation_mask, 
                            self.current_image.width(),
                            self.current_image.height()
                        )
                    else:
                        print(f"[ScreenshotPreviewWidget] splitter未找到或没有on_segmentation_mask_ready方法")
                else:
                    print(f"[ScreenshotPreviewWidget] 没有_splitter_ref，无法通知splitter")
            else:
                self.segment_status_label.setText("√ 分割成功，但未检测到对象")
                self.segment_status_label.setStyleSheet("color: #FFD93D; font-size: 10px;")
        else:
            error_msg = result.get('summary', '分割失败')
            self.segment_status_label.setText(f"✗ {error_msg}")
            self.segment_status_label.setStyleSheet("color: #FF6B6B; font-size: 10px;")
    
    def _geojson_to_mask(self, geojson: dict, width: int, height: int) -> np.ndarray:
        """将GeoJSON转换为2D mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if 'features' not in geojson:
            return mask
        
        try:
            for idx, feature in enumerate(geojson['features']):
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'Polygon':
                    coordinates = geometry.get('coordinates', [])
                    if coordinates and len(coordinates) > 0:
                        # 获取第一个环（外环）
                        ring = coordinates[0]
                        # 转换为OpenCV格式的轮廓点
                        points = np.array([[int(p[0]), int(p[1])] for p in ring], dtype=np.int32)
                        # 在mask上绘制多边形（使用不同的标签值）
                        cv2.fillPoly(mask, [points], idx + 1)
        except Exception as e:
            print(f"转换GeoJSON到mask失败: {e}")
        
        return mask
    
    def set_sam3_api_url(self, url: str):
        """设置SAM3 API地址"""
        self.sam3_api_url = url
    
    def update_info(self, width: int, height: int, start_3d: Tuple[float, float, float], 
                   end_3d: Tuple[float, float, float]):
        """
        更新信息显示
        
        Args:
            width: 图像宽度
            height: 图像高度
            start_3d: 起点3D坐标（可能是本地坐标或地理坐标）
            end_3d: 终点3D坐标（可能是本地坐标或地理坐标）
        """
        info_text = f"尺寸: {width} × {height} 像素\n"
        # 判断是地理坐标还是本地坐标（地理坐标的经度通常在-180到180之间）
        if abs(start_3d[0]) <= 180 and abs(start_3d[1]) <= 90:
            # 地理坐标格式
            info_text += f"起点(地理): 经度={start_3d[0]:.6f}°, 纬度={start_3d[1]:.6f}°, 高度={start_3d[2]:.2f}m\n"
            info_text += f"终点(地理): 经度={end_3d[0]:.6f}°, 纬度={end_3d[1]:.6f}°, 高度={end_3d[2]:.2f}m"
        else:
            # 本地坐标格式
            info_text += f"起点(3D): ({start_3d[0]:.2f}, {start_3d[1]:.2f}, {start_3d[2]:.2f})\n"
            info_text += f"终点(3D): ({end_3d[0]:.2f}, {end_3d[1]:.2f}, {end_3d[2]:.2f})"
        self.info_label.setText(info_text)


class RayCaster:
    """射线检测器，用于像素坐标转3D坐标"""
    
    def __init__(self, gaussians, pipeline, background):
        """
        初始化射线检测器
        
        Args:
            gaussians: GaussianModel对象
            pipeline: 渲染管道
            background: 背景颜色
        """
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
    
    def pixel_to_ray(self, pixel_x: float, pixel_y: float, 
                     view_matrix: np.ndarray, proj_matrix: np.ndarray,
                     image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        将像素坐标转换为射线（起点和方向）
        
        Args:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            view_matrix: 视图矩阵 (4x4)
            proj_matrix: 投影矩阵 (4x4)
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            (ray_origin, ray_direction): 射线起点和方向向量
        """
        # 将像素坐标转换为NDC坐标 (-1到1)
        ndc_x = (2.0 * pixel_x / image_width) - 1.0
        ndc_y = 1.0 - (2.0 * pixel_y / image_height)  # Y轴翻转
        
        # 计算射线方向（在视图空间中）
        # 使用投影矩阵的逆矩阵将NDC坐标转换为视图空间坐标
        proj_inv = np.linalg.inv(proj_matrix)
        
        # 近平面和远平面的NDC坐标
        near_point_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_point_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])
        
        # 转换为视图空间
        near_point_view = proj_inv @ near_point_ndc
        near_point_view /= near_point_view[3]
        
        far_point_view = proj_inv @ far_point_ndc
        far_point_view /= far_point_view[3]
        
        # 计算射线方向（在视图空间中）
        ray_dir_view = far_point_view[:3] - near_point_view[:3]
        ray_dir_view = ray_dir_view / np.linalg.norm(ray_dir_view)
        
        # 射线起点是相机位置（视图空间原点）
        ray_origin_view = np.array([0.0, 0.0, 0.0])
        
        # 转换为世界空间
        view_inv = np.linalg.inv(view_matrix)
        
        # 转换起点
        ray_origin_homogeneous = np.array([*ray_origin_view, 1.0])
        ray_origin_world = (view_inv @ ray_origin_homogeneous)[:3]
        
        # 转换方向（只旋转，不平移）
        ray_dir_homogeneous = np.array([*ray_dir_view, 0.0])
        ray_dir_world = (view_inv @ ray_dir_homogeneous)[:3]
        ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
        
        return ray_origin_world, ray_dir_world
    
    def ray_intersect_gaussians(self, ray_origin: np.ndarray, ray_direction: np.ndarray,
                               max_distance: float = 100.0) -> Optional[np.ndarray]:
        """
        使用GaussianSplats3D库的算法进行射线与高斯点的相交检测
        
        实现基于GaussianSplats3D库的Raycaster.intersectSplatMesh方法：
        1. 对每个高斯点，计算射线与高斯椭球体的相交
        2. 高斯点被视为椭球体，使用射线-球体相交检测（简化版）
        3. 返回最近的交点
        
        Args:
            ray_origin: 射线起点 (3,)
            ray_direction: 射线方向 (3,)
            max_distance: 最大检测距离
            
        Returns:
            交点3D坐标，如果没有找到则返回None
        """
        if self.gaussians is None:
            return None
        
        # 获取所有高斯点的数据
        gaussian_positions = self.gaussians.get_xyz.detach().cpu().numpy()
        
        if len(gaussian_positions) == 0:
            return None
        
        # 获取缩放（用于计算椭球体半径）
        scales = self.gaussians.get_scaling.detach().cpu().numpy()
        
        ray_origin = np.array(ray_origin)
        ray_direction = np.array(ray_direction)
        ray_direction = ray_direction / np.linalg.norm(ray_direction)  # 归一化
        
        # 存储所有相交点
        hits = []
        
        # 对每个高斯点进行射线-椭球体相交检测
        # 根据GaussianSplats3D的实现，使用简化的球体相交检测
        for i in range(len(gaussian_positions)):
            center = gaussian_positions[i]
            scale = scales[i]
            
            # 计算椭球体的平均半径（类似GaussianSplats3D的实现）
            # 在3D模式下：radius = (scale.x + scale.y + scale.z) / 3
            # 在2D模式下：radius = (scale.x + scale.y) / 2
            # 这里使用3D模式
            if len(scale) >= 3:
                radius = (scale[0] + scale[1] + scale[2]) / 3.0
            elif len(scale) >= 2:
                radius = (scale[0] + scale[1]) / 2.0
            else:
                radius = scale[0] if len(scale) > 0 else 0.01
            
            # 射线-球体相交检测
            # 算法：计算从射线起点到球心的向量，然后计算射线到球心的距离
            oc = center - ray_origin
            proj_length = np.dot(oc, ray_direction)
            
            # 只考虑射线前方的点
            if proj_length < 0:
                continue
            
            # 计算射线到球心的最短距离
            oc_proj = ray_origin + proj_length * ray_direction
            dist_to_center = np.linalg.norm(center - oc_proj)
            
            # 如果距离小于半径，则相交
            if dist_to_center <= radius:
                # 计算交点（射线与球面的交点）
                # 使用勾股定理：从投影点到交点的距离
                half_chord = np.sqrt(radius ** 2 - dist_to_center ** 2)
                
                # 计算两个可能的交点，选择较近的（在射线方向上）
                hit_point = oc_proj - half_chord * ray_direction
                
                # 计算交点到射线起点的距离
                hit_distance = np.linalg.norm(hit_point - ray_origin)
                
                if hit_distance <= max_distance:
                    hits.append({
                        'point': hit_point,
                        'distance': hit_distance,
                        'splat_index': i
                    })
        
        if len(hits) == 0:
            return None
        
        # 按距离排序，返回最近的交点
        hits.sort(key=lambda x: x['distance'])
        return hits[0]['point']
    
    def pixel_to_3d_with_ground_constraint(self, pixel_x: float, pixel_y: float,
                                          view_matrix: np.ndarray, proj_matrix: np.ndarray,
                                          image_width: int, image_height: int,
                                          ground_y: float = 0.0) -> Optional[np.ndarray]:
        """
        将像素坐标转换为3D坐标，并约束到XZ平面（贴地效果）
        
        Args:
            pixel_x: 像素X坐标
            pixel_y: 像素Y坐标
            view_matrix: 视图矩阵
            proj_matrix: 投影矩阵
            image_width: 图像宽度
            image_height: 图像高度
            ground_y: 地面Y坐标（默认0，即XZ平面）
            
        Returns:
            3D坐标 (x, y, z)，y被约束为ground_y
        """
        # 获取射线
        ray_origin, ray_direction = self.pixel_to_ray(
            pixel_x, pixel_y, view_matrix, proj_matrix, image_width, image_height
        )
        
        # 计算射线与XZ平面（y = ground_y）的交点
        # 射线方程：P = O + t * D
        # 平面方程：y = ground_y
        # 求解：O_y + t * D_y = ground_y
        # t = (ground_y - O_y) / D_y
        
        if abs(ray_direction[1]) < 1e-6:
            # 射线平行于XZ平面，无法相交
            return None
        
        t = (ground_y - ray_origin[1]) / ray_direction[1]
        
        if t < 0:
            # 交点在射线后方，不可用
            return None
        
        # 计算交点
        intersection = ray_origin + t * ray_direction
        
        return intersection


class GSScreenshotSplitter:
    """3DGS截图分割器"""
    
    def __init__(self, gaussians=None, pipeline=None, background=None, viewer_widget=None, cesium_widget=None):
        """
        初始化截图分割器
        
        Args:
            gaussians: GaussianModel对象（可选，如果为None则使用Cesium模式）
            pipeline: 渲染管道（可选）
            background: 背景颜色（可选）
            viewer_widget: 3D查看器widget（GaussianMultiModeRenderer，可选）
            cesium_widget: Cesium widget（可选，用于Cesium模式）
        """
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.viewer_widget = viewer_widget
        self.cesium_widget = cesium_widget
        
        # 判断使用哪种模式
        self.use_cesium_mode = (gaussians is None and cesium_widget is not None)
        
        if not self.use_cesium_mode and gaussians is not None:
            self.ray_caster = RayCaster(gaussians, pipeline, background)
        else:
            self.ray_caster = None  # Cesium模式下，射线检测在JavaScript端进行
        
        # 截图相关状态
        self.current_screenshot = None
        self.corner_points_2d = []  # 四个角点的2D像素坐标
        self.corner_points_3d = []   # 四个角点的3D坐标（约束到XZ平面）
        self.corner_points_geo = []  # 四个角点的地理坐标（Cesium模式）
        
        # 预览面板
        self.preview_widget = None
        
        # 异步操作等待标志
        self._waiting_for_screenshot = False
        self._waiting_for_corner_points = False
        self._screenshot_result = None
        self._corner_points_result = None
    
    def capture_screenshot(self, width: int = None, height: int = None) -> Optional[QImage]:
        """
        捕获当前视角的截图
        
        Args:
            width: 图像宽度（如果为None则使用查看器宽度）
            height: 图像高度（如果为None则使用查看器高度）
            
        Returns:
            截图图像，如果失败则返回None
        """
        if self.use_cesium_mode:
            # Cesium模式：从Cesium视图获取截图
            return self._capture_cesium_screenshot(width, height)
        else:
            # Python 3D查看器模式
            if self.viewer_widget is None:
                return None
            
            try:
                # 获取查看器尺寸
                if width is None:
                    width = self.viewer_widget.view.width()
                if height is None:
                    height = self.viewer_widget.view.height()
                
                # 从pyqtgraph GL视图获取截图
                # pyqtgraph的GLViewWidget有grabFramebuffer方法
                if hasattr(self.viewer_widget.view, 'grabFramebuffer'):
                    pixmap = self.viewer_widget.view.grabFramebuffer()
                    image = pixmap.toImage()
                else:
                    # 备用方法：如果grabFramebuffer不可用，尝试使用QWidget的grab方法
                    pixmap = self.viewer_widget.view.grab()
                    image = pixmap.toImage()
                
                self.current_screenshot = image
                return image
                
            except Exception as e:
                print(f"截图捕获失败: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def _capture_cesium_screenshot(self, width: int = None, height: int = None) -> Optional[QImage]:
        """
        从Cesium视图捕获截图（合并Cesium和Three.js的canvas，不包含UI面板）
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            截图图像，如果失败则返回None
        """
        if self.cesium_widget is None:
            return None
        
        try:
            # 通过信号机制等待JavaScript端的响应
            from PyQt5.QtCore import QEventLoop, QTimer, QByteArray
            import time
            
            # 先等待一小段时间确保页面已加载
            time.sleep(0.2)  # 等待200ms
            
            # 重试机制：最多尝试3次
            max_retries = 3
            for attempt in range(max_retries):
                self._waiting_for_screenshot = True
                self._screenshot_result = None
                
                # 连接信号
                bridge = self.cesium_widget.bridge
                bridge.screenshot_captured.connect(self._on_screenshot_captured)
                
                # 发送截图请求
                message = {
                    'type': 'captureScreenshot',
                    'data': {
                        'width': width,
                        'height': height
                    }
                }
                self.cesium_widget.send_message(message)
                
                # 等待结果（最多等待5秒）
                loop = QEventLoop()
                timer = QTimer()
                timer.setSingleShot(True)
                timeout_timer = QTimer()
                timeout_timer.setSingleShot(True)
                
                def check_result():
                    if not self._waiting_for_screenshot:
                        loop.quit()
                    else:
                        timer.start(50)  # 50ms后重试
                
                def timeout():
                    self._waiting_for_screenshot = False
                    loop.quit()
                
                timer.timeout.connect(check_result)
                timeout_timer.timeout.connect(timeout)
                timeout_timer.start(5000)  # 5秒超时
                timer.start(50)
                loop.exec_()
                
                # 断开信号连接
                try:
                    bridge.screenshot_captured.disconnect(self._on_screenshot_captured)
                except:
                    pass
                
                if self._screenshot_result:
                    return self._screenshot_result
                
                # 如果失败且不是最后一次尝试，等待一下再重试
                if attempt < max_retries - 1:
                    time.sleep(0.3)  # 等待300ms后重试
                    print(f"[INFO] 截图捕获失败，重试 {attempt + 2}/{max_retries}")
            
            # 如果所有重试都失败，使用备用方法
            print("[WARNING] 异步截图失败，使用备用方法（可能包含UI元素）")
            if hasattr(self.cesium_widget, 'web_view'):
                pixmap = self.cesium_widget.web_view.grab()
                image = pixmap.toImage()
                self.current_screenshot = image
                return image
            
            return None
            
        except Exception as e:
            print(f"Cesium截图捕获失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _on_screenshot_captured(self, data: dict):
        """处理截图捕获完成信号"""
        try:
            from PyQt5.QtCore import QByteArray
            image_data_str = data.get('data', {}).get('imageData', '')
            
            if image_data_str and image_data_str.startswith('data:image'):
                # 移除data:image/png;base64,前缀
                base64_data = image_data_str.split(',')[1]
                image_data = QByteArray.fromBase64(base64_data.encode())
                image = QImage.fromData(image_data, 'PNG')
                if not image.isNull():
                    self._screenshot_result = image
                    self.current_screenshot = image
                    self._waiting_for_screenshot = False
        except Exception as e:
            print(f"处理截图数据失败: {e}")
            self._waiting_for_screenshot = False
    
    def get_camera_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取当前相机的视图矩阵和投影矩阵
        
        Returns:
            (view_matrix, proj_matrix): 视图矩阵和投影矩阵，如果失败则返回(None, None)
        """
        if self.use_cesium_mode:
            # Cesium模式：从Cesium获取相机参数
            return self._get_cesium_camera_matrices()
        else:
            # Python 3D查看器模式
            if self.viewer_widget is None:
                return None, None
            
            try:
                # 从pyqtgraph GL视图获取矩阵
                view_matrix = self.viewer_widget.view.viewMatrix()
                proj_matrix = self.viewer_widget.view.projectionMatrix()
                
                # 转换为numpy数组
                view_np = np.array(view_matrix.data()).reshape(4, 4).T
                proj_np = np.array(proj_matrix.data()).reshape(4, 4).T
                
                return view_np, proj_np
                
            except Exception as e:
                print(f"获取相机矩阵失败: {e}")
                return None, None
    
    def _get_cesium_camera_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从Cesium获取相机矩阵
        
        Returns:
            (view_matrix, proj_matrix): 视图矩阵和投影矩阵
        """
        if self.cesium_widget is None:
            return None, None
        
        try:
            # 通过JavaScript获取Cesium的相机参数
            # 发送消息到JavaScript端请求相机矩阵
            message = {
                'type': 'getCameraMatrices'
            }
            self.cesium_widget.send_message(message)
            
            # 注意：这里需要等待JavaScript端的响应
            # 由于是异步的，我们需要通过信号/回调机制获取结果
            # TODO: 实现异步相机矩阵获取机制
            
            # 临时返回None，实际实现需要通过信号机制
            return None, None
            
        except Exception as e:
            print(f"获取Cesium相机矩阵失败: {e}")
            return None, None
    
    def calculate_corner_points_3d(self, image_width: int, image_height: int,
                                   corner_points_2d: List[QPoint] = None) -> List[np.ndarray]:
        """
        计算四个角点的3D坐标（约束到XZ平面）
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            corner_points_2d: 四个角点的2D像素坐标，如果为None则使用图像四个角
            
        Returns:
            四个角点的3D坐标列表
        """
        if self.use_cesium_mode:
            # Cesium模式：通过JavaScript使用GaussianSplats3D的Raycaster
            return self._calculate_cesium_corner_points_3d(image_width, image_height, corner_points_2d)
        else:
            # Python模式：使用Python端的射线检测
            # 获取相机矩阵
            view_matrix, proj_matrix = self.get_camera_matrices()
            if view_matrix is None or proj_matrix is None:
                return []
            
            # 如果没有提供2D角点，使用图像四个角
            if corner_points_2d is None or len(corner_points_2d) != 4:
                corner_points_2d = [
                    QPoint(0, 0),  # 左上
                    QPoint(image_width - 1, 0),  # 右上
                    QPoint(image_width - 1, image_height - 1),  # 右下
                    QPoint(0, image_height - 1)  # 左下
                ]
            
            self.corner_points_2d = corner_points_2d
            
            # 计算每个角点的3D坐标（约束到XZ平面，即Y=0）
            corner_points_3d = []
            for point_2d in corner_points_2d:
                point_3d = self.ray_caster.pixel_to_3d_with_ground_constraint(
                    point_2d.x(), point_2d.y(),
                    view_matrix, proj_matrix,
                    image_width, image_height,
                    ground_y=0.0  # XZ平面，Y=0
                )
                
                if point_3d is not None:
                    corner_points_3d.append(point_3d)
                else:
                    # 如果无法计算，使用默认值
                    corner_points_3d.append(np.array([0.0, 0.0, 0.0]))
            
            self.corner_points_3d = corner_points_3d
            return corner_points_3d
    
    def _calculate_cesium_corner_points_3d(self, image_width: int, image_height: int,
                                          corner_points_2d: List[QPoint] = None) -> List[np.ndarray]:
        """
        在Cesium模式下计算四个角点的3D坐标（通过JavaScript将射线与Cesium地球表面求交）
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            corner_points_2d: 四个角点的2D像素坐标
            
        Returns:
            四个角点的3D坐标列表（本地坐标）
        """
        if self.cesium_widget is None:
            return []
        
        # 如果没有提供2D角点，使用图像四个角
        if corner_points_2d is None or len(corner_points_2d) != 4:
            corner_points_2d = [
                QPoint(0, 0),  # 左上
                QPoint(image_width - 1, 0),  # 右上
                QPoint(image_width - 1, image_height - 1),  # 右下
                QPoint(0, image_height - 1)  # 左下
            ]
        
        self.corner_points_2d = corner_points_2d
        
        try:
            # 通过信号机制等待JavaScript端的响应
            from PyQt5.QtCore import QEventLoop, QTimer
            
            self._waiting_for_corner_points = True
            self._corner_points_result = None
            
            # 连接信号
            bridge = self.cesium_widget.bridge
            bridge.corner_points_calculated.connect(self._on_corner_points_calculated)
            
            # 发送计算请求
            pixel_points = [
                {'x': p.x(), 'y': p.y()} for p in corner_points_2d
            ]
            
            message = {
                'type': 'calculateCornerPoints3D',
                'data': {
                    'imageWidth': image_width,
                    'imageHeight': image_height,
                    'pixelPoints': pixel_points
                }
            }
            self.cesium_widget.send_message(message)
            
            # 等待结果（最多等待3秒）
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)
            timeout_timer = QTimer()
            timeout_timer.setSingleShot(True)
            
            def check_result():
                if not self._waiting_for_corner_points:
                    loop.quit()
                else:
                    timer.start(50)  # 50ms后重试
            
            def timeout():
                self._waiting_for_corner_points = False
                loop.quit()
            
            timer.timeout.connect(check_result)
            timeout_timer.timeout.connect(timeout)
            timeout_timer.start(3000)  # 3秒超时
            timer.start(50)
            loop.exec_()
            
            # 断开信号连接
            try:
                bridge.corner_points_calculated.disconnect(self._on_corner_points_calculated)
            except:
                pass
            
            if self._corner_points_result:
                # 检查结果类型
                if isinstance(self._corner_points_result[0], np.ndarray):
                    # Three.js坐标（XZ平面上的点），直接返回
                    self.corner_points_3d = self._corner_points_result
                    return self._corner_points_result
                else:
                    # Cesium地理坐标，存储在corner_points_geo中
                    self.corner_points_geo = self._corner_points_result
                    # 暂时返回空数组，实际坐标在corner_points_geo中
                    return [np.array([0.0, 0.0, 0.0]) for _ in range(4)]
            
            # 如果失败，返回默认值
            return [np.array([0.0, 0.0, 0.0]) for _ in range(4)]
            
        except Exception as e:
            print(f"Cesium角点计算失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _on_corner_points_calculated(self, data: dict):
        """处理角点计算完成信号"""
        try:
            data_dict = data.get('data', {})
            corner_points = data_dict.get('cornerPoints', [])
            corner_points_three = data_dict.get('cornerPointsThree', [])
            
            # 优先使用Three.js坐标（XZ平面上的点）
            if corner_points_three and len(corner_points_three) == 4:
                # 将Three.js坐标转换为numpy数组
                self._corner_points_result = [
                    np.array([pt['x'], pt['y'], pt['z']]) for pt in corner_points_three
                ]
                self._waiting_for_corner_points = False
                print(f"[ScreenshotSplitter] 接收到Three.js坐标（XZ平面）: {self._corner_points_result}")
            elif corner_points and len(corner_points) == 4:
                # 如果没有Three.js坐标，使用Cesium地理坐标
                self._corner_points_result = corner_points
                self._waiting_for_corner_points = False
                print(f"[ScreenshotSplitter] 接收到Cesium地理坐标: {self._corner_points_result}")
            else:
                print(f"[ScreenshotSplitter] 警告: 角点数据不完整，cornerPoints={len(corner_points)}, cornerPointsThree={len(corner_points_three) if corner_points_three else 0}")
                self._waiting_for_corner_points = False
        except Exception as e:
            print(f"处理角点数据失败: {e}")
            import traceback
            traceback.print_exc()
            self._waiting_for_corner_points = False
    
    def create_preview_widget(self) -> ScreenshotPreviewWidget:
        """创建预览面板"""
        if self.preview_widget is None:
            self.preview_widget = ScreenshotPreviewWidget()
            # 设置父对象，以便preview_widget可以访问splitter的方法
            # 注意：这里不能直接设置parent，因为preview_widget可能在不同的窗口
            # 我们通过一个弱引用来避免循环引用
            import weakref
            self.preview_widget._splitter_ref = weakref.ref(self)
        return self.preview_widget
    
    def update_preview(self):
        """更新预览面板"""
        if self.preview_widget is None or self.current_screenshot is None:
            return
        
        # 计算角点3D坐标
        width = self.current_screenshot.width()
        height = self.current_screenshot.height()
        
        corner_points_3d = self.calculate_corner_points_3d(width, height)
        
        # 根据模式选择显示的坐标
        if self.use_cesium_mode and hasattr(self, 'corner_points_geo') and len(self.corner_points_geo) >= 4:
            # Cesium模式：使用地理坐标
            start_geo = self.corner_points_geo[0]
            end_geo = self.corner_points_geo[2] if len(self.corner_points_geo) > 2 else self.corner_points_geo[1]
            start_3d = (start_geo.get('longitude', 0.0), start_geo.get('latitude', 0.0), start_geo.get('height', 0.0))
            end_3d = (end_geo.get('longitude', 0.0), end_geo.get('latitude', 0.0), end_geo.get('height', 0.0))
        elif len(corner_points_3d) >= 2:
            start_3d = corner_points_3d[0]
            end_3d = corner_points_3d[2] if len(corner_points_3d) > 2 else corner_points_3d[1]
        else:
            start_3d = (0.0, 0.0, 0.0)
            end_3d = (0.0, 0.0, 0.0)
        
        # 更新预览
        self.preview_widget.set_preview_image(
            self.current_screenshot, 
            self.corner_points_2d
        )
        self.preview_widget.update_info(
            width, height, 
            tuple(start_3d), 
            tuple(end_3d)
        )
    
    def download_screenshot(self, file_path: str = None) -> str:
        """
        下载截图（包含分割结果）
        
        Args:
            file_path: 保存路径，如果为None则弹出文件对话框
            
        Returns:
            保存的文件路径
        """
        if self.current_screenshot is None:
            return ""
        
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "保存截图",
                "",
                "PNG图像 (*.png);;JPEG图像 (*.jpg);;所有文件 (*)"
            )
            if not file_path:
                return ""
        
        # 创建带分割结果的完整图像
        pixmap = QPixmap.fromImage(self.current_screenshot)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 从预览面板获取分割结果
        segmentation_mask = None
        corner_points = []
        if self.preview_widget is not None:
            segmentation_mask = self.preview_widget.segmentation_mask
            corner_points = self.preview_widget.corner_points
        
        # 绘制分割结果
        if segmentation_mask is not None:
            self._draw_segmentation_mask_on_pixmap(painter, segmentation_mask, pixmap.width(), pixmap.height())
        
        # 绘制角点标记
        if corner_points and len(corner_points) == 4:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            brush = QBrush(QColor(255, 0, 0, 100))
            painter.setBrush(brush)
            
            radius = 15
            for i, point in enumerate(corner_points):
                if i == 0:  # 左上角
                    painter.drawPie(point.x() - radius, point.y() - radius, 
                                   radius * 2, radius * 2, 90 * 16, 90 * 16)
                elif i == 1:  # 右上角
                    painter.drawPie(point.x() - radius, point.y() - radius, 
                                   radius * 2, radius * 2, 0, 90 * 16)
                elif i == 2:  # 右下角
                    painter.drawPie(point.x() - radius, point.y() - radius, 
                                   radius * 2, radius * 2, 270 * 16, 90 * 16)
                elif i == 3:  # 左下角
                    painter.drawPie(point.x() - radius, point.y() - radius, 
                                   radius * 2, radius * 2, 180 * 16, 90 * 16)
        
        painter.end()
        
        # 保存图像
        success = pixmap.save(file_path)
        if not success:
            # 如果保存失败，尝试使用QImage保存
            image = pixmap.toImage()
            success = image.save(file_path)
        
        return file_path if success else ""
    
    def _draw_segmentation_mask_on_pixmap(self, painter: QPainter, mask: np.ndarray, image_width: int, image_height: int):
        """在pixmap上绘制分割mask"""
        if mask is None:
            return
        
        try:
            # 将mask缩放到图像尺寸
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 为每个检测到的对象使用不同颜色
            unique_labels = np.unique(mask_resized)
            colors = [
                QColor(255, 0, 0, 150),    # 红色
                QColor(0, 255, 0, 150),    # 绿色
                QColor(0, 0, 255, 150),    # 蓝色
                QColor(255, 255, 0, 150),  # 黄色
                QColor(255, 0, 255, 150),  # 洋红
                QColor(0, 255, 255, 150),  # 青色
            ]
            
            for idx, label in enumerate(unique_labels):
                if label == 0:  # 背景
                    continue
                
                color = colors[idx % len(colors)]
                brush = QBrush(color)
                painter.setBrush(brush)
                painter.setPen(Qt.NoPen)
                
                # 找到该标签的轮廓
                mask_label = (mask_resized == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 绘制每个轮廓
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    # 转换为QPolygon
                    points = [QPoint(int(p[0][0]), int(p[0][1])) for p in contour]
                    polygon = QPolygon(points)
                    painter.drawPolygon(polygon)
        except Exception as e:
            print(f"绘制分割mask失败: {e}")
    
    def get_corner_points_3d(self) -> List[np.ndarray]:
        """获取四个角点的3D坐标"""
        return self.corner_points_3d.copy() if self.corner_points_3d else []
    
    def get_segmentation_mask(self) -> Optional[np.ndarray]:
        """获取分割mask"""
        return self.segmentation_mask.copy() if self.segmentation_mask is not None else None
    
    def get_segmentation_result(self) -> Optional[dict]:
        """获取分割结果（GeoJSON）"""
        return self.segmentation_result.copy() if self.segmentation_result is not None else None
    
    def on_segmentation_mask_ready(self, mask: np.ndarray, image_width: int, image_height: int):
        """
        当分割mask准备好时调用（由ScreenshotPreviewWidget触发）
        
        Args:
            mask: 分割mask
            image_width: 图像宽度
            image_height: 图像高度
        """
        print(f"[ScreenshotSplitter] on_segmentation_mask_ready被调用，use_cesium_mode={self.use_cesium_mode}, cesium_widget={self.cesium_widget is not None}")
        if self.use_cesium_mode and self.cesium_widget:
            # 延迟一点时间，确保mask已经设置
            from PyQt5.QtCore import QTimer
            print(f"[ScreenshotSplitter] 准备计算轮廓3D坐标，mask尺寸: {mask.shape if mask is not None else 'None'}, 图像尺寸: {image_width}x{image_height}")
            QTimer.singleShot(200, lambda: self.calculate_contour_points_3d(image_width, image_height, mask))
        else:
            print(f"[ScreenshotSplitter] 跳过轮廓3D坐标计算: use_cesium_mode={self.use_cesium_mode}, cesium_widget={self.cesium_widget is not None}")
    
    def calculate_contour_points_3d(self, image_width: int, image_height: int, mask: np.ndarray = None) -> bool:
        """
        计算分割轮廓的3D坐标（约束到XZ平面）
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            是否成功发送计算请求
        """
        if self.cesium_widget is None:
            print("[ScreenshotSplitter] Cesium widget未初始化")
            return False
        
        # 使用传入的mask，如果没有则使用self.segmentation_mask
        segmentation_mask = mask if mask is not None else self.segmentation_mask
        
        if segmentation_mask is None:
            print("[ScreenshotSplitter] 没有分割mask数据")
            return False
        
        try:
            print(f"[ScreenshotSplitter] 开始提取轮廓点，mask类型: {type(segmentation_mask)}, mask形状: {segmentation_mask.shape if hasattr(segmentation_mask, 'shape') else 'N/A'}")
            
            # 提取轮廓点
            contours = self._extract_contour_points(segmentation_mask, image_width, image_height)
            
            print(f"[ScreenshotSplitter] 提取到{len(contours)}个轮廓")
            
            if not contours or len(contours) == 0:
                print("[ScreenshotSplitter] 未找到轮廓")
                return False
            
            # 转换为像素坐标点数组
            contour_points = []
            for idx, contour in enumerate(contours):
                pixel_points = [{'x': int(p.x()), 'y': int(p.y())} for p in contour]
                contour_points.append(pixel_points)
                print(f"[ScreenshotSplitter] 轮廓{idx}: {len(pixel_points)}个点")
            
            total_points = sum(len(c) for c in contour_points)
            print(f"[ScreenshotSplitter] 提取到{len(contour_points)}个轮廓，共{total_points}个点")
            
            # 发送计算请求到JavaScript
            message = {
                'type': 'calculateContourPoints3D',
                'data': {
                    'imageWidth': image_width,
                    'imageHeight': image_height,
                    'contourPoints': contour_points,
                    'contourId': id(self)  # 使用对象ID作为轮廓ID，用于区分不同的分割结果
                }
            }
            print(f"[ScreenshotSplitter] 发送轮廓计算请求到JavaScript，轮廓数: {len(contour_points)}")
            self.cesium_widget.send_message(message)
            
            return True
            
        except Exception as e:
            print(f"[ScreenshotSplitter] 计算轮廓3D坐标失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_contour_points(self, mask: np.ndarray, image_width: int, image_height: int) -> List[List[QPoint]]:
        """
        从分割mask中提取轮廓点
        
        Args:
            mask: 分割mask（numpy数组）
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            轮廓点列表，每个轮廓是一个QPoint列表
        """
        try:
            # 将mask缩放到图像尺寸
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 获取所有唯一标签
            unique_labels = np.unique(mask_resized)
            unique_labels = unique_labels[unique_labels > 0]  # 排除背景（0）
            
            all_contours = []
            
            # 为每个标签提取轮廓
            for label in unique_labels:
                # 找到该标签的轮廓
                mask_label = (mask_resized == label).astype(np.uint8)
                contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 转换每个轮廓为QPoint列表
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    # 转换为QPoint列表
                    points = [QPoint(int(p[0][0]), int(p[0][1])) for p in contour]
                    all_contours.append(points)
            
            return all_contours
            
        except Exception as e:
            print(f"[ScreenshotSplitter] 提取轮廓点失败: {e}")
            import traceback
            traceback.print_exc()
            return []


def call_sam3_api(image: QImage, prompt: str, api_url: str = "http://localhost:5000", 
                  threshold: float = 0.5, mask_threshold: float = 0.5) -> dict:
    """
    调用SAM3 API进行图像分割
    
    Args:
        image: QImage对象
        prompt: 文本提示词（如 "building", "car" 等）
        api_url: SAM3 API服务器地址
        threshold: 置信度阈值
        mask_threshold: mask阈值
    
    Returns:
        包含分割结果的字典，格式：
        {
            'success': bool,
            'geojson': dict,  # GeoJSON格式的分割结果
            'summary': str    # 摘要信息
        }
    """
    try:
        # 将QImage转换为PIL Image
        qimage = image
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA
        rgb_arr = arr[:, :, :3]  # 只取RGB
        pil_image = Image.fromarray(rgb_arr)
        
        # 将PIL Image转换为base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 准备请求数据
        data = {
            'image': image_b64,
            'prompt': prompt,
            'threshold': threshold,
            'mask_threshold': mask_threshold
        }
        
        # 发送POST请求
        api_endpoint = f"{api_url}/api/segment"
        response = requests.post(api_endpoint, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                'success': False,
                'summary': f"API请求失败: HTTP {response.status_code} - {response.text}"
            }
    
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'summary': f"无法连接到SAM3 API服务器 ({api_url})，请确保服务器正在运行"
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'summary': "API请求超时，请检查网络连接或服务器响应速度"
        }
    except Exception as e:
        return {
            'success': False,
            'summary': f"分割失败: {str(e)}"
        }

