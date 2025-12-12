#!/usr/bin/env python3
"""
Three.js 3DGS分割视图
独立的Three.js视图标签页，用于3DGS模型的可视化、截图、SAM3分割和高斯点选择操作
与Cesium GIS视图分离，专注于分割工作流
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject, Qt, QTimer, QEventLoop
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QFrame, QGroupBox, QLineEdit, QDoubleSpinBox,
    QProgressBar, QSplitter, QFileDialog, QComboBox, QCheckBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel


class ThreeJSBridge(QObject):
    """Python和Three.js JavaScript之间的桥接对象"""
    
    # 定义信号
    message_received = pyqtSignal(str)
    viewer_ready = pyqtSignal()
    model_loaded = pyqtSignal(bool, str)  # (success, message)
    screenshot_captured = pyqtSignal(dict)
    segmentation_complete = pyqtSignal(dict)
    points_selected = pyqtSignal(dict)
    model_split_complete = pyqtSignal(dict)
    # 新增编辑相关信号
    select_by_box = pyqtSignal(dict)
    select_by_lasso = pyqtSignal(dict)
    select_by_brush = pyqtSignal(dict)
    select_all = pyqtSignal()
    select_inverse = pyqtSignal()
    clear_selection = pyqtSignal()
    hide_selected = pyqtSignal()
    show_all = pyqtSignal()
    lock_selected = pyqtSignal()
    unlock_all = pyqtSignal()
    delete_selected = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    render_mode_changed = pyqtSignal(str)
    transform_mode_changed = pyqtSignal(str)
    # 场景管理器相关信号
    model_selected = pyqtSignal(str)  # model id
    model_deselected = pyqtSignal()
    model_deleted = pyqtSignal(str)  # model id
    model_transformed = pyqtSignal(dict)  # transform data
    request_add_model = pyqtSignal()
    request_point_cloud_overlay = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    @pyqtSlot(str)
    def receiveMessage(self, message: str):
        """从JavaScript接收消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            print(f"[ThreeJSBridge] 收到消息: {msg_type}")
            
            if msg_type == 'viewerReady':
                self.viewer_ready.emit()
            elif msg_type == 'modelLoaded':
                success = data.get('success', False)
                error = data.get('error', '')
                self.model_loaded.emit(success, error)
            elif msg_type == 'screenshotCaptured':
                self.screenshot_captured.emit(data)
            elif msg_type == 'segmentationComplete':
                self.segmentation_complete.emit(data)
            elif msg_type == 'pointsSelected':
                self.points_selected.emit(data)
            elif msg_type == 'modelSplitComplete':
                self.model_split_complete.emit(data)
            # 新增编辑消息处理
            elif msg_type == 'selectByBox':
                self.select_by_box.emit(data.get('data', {}))
            elif msg_type == 'selectByLasso':
                self.select_by_lasso.emit(data.get('data', {}))
            elif msg_type == 'selectByBrush':
                self.select_by_brush.emit(data.get('data', {}))
            elif msg_type == 'selectAll':
                self.select_all.emit()
            elif msg_type == 'selectInverse':
                self.select_inverse.emit()
            elif msg_type == 'clearSelection':
                self.clear_selection.emit()
            elif msg_type == 'hideSelected':
                self.hide_selected.emit()
            elif msg_type == 'showAll':
                self.show_all.emit()
            elif msg_type == 'lockSelected':
                self.lock_selected.emit()
            elif msg_type == 'unlockAll':
                self.unlock_all.emit()
            elif msg_type == 'deleteSelected':
                self.delete_selected.emit()
            elif msg_type == 'undo':
                self.undo_requested.emit()
            elif msg_type == 'redo':
                self.redo_requested.emit()
            elif msg_type == 'renderModeChanged':
                self.render_mode_changed.emit(data.get('data', {}).get('mode', 'splat'))
            elif msg_type == 'transformModeChanged':
                self.transform_mode_changed.emit(data.get('data', {}).get('mode', ''))
            # 场景管理器相关消息
            elif msg_type == 'modelSelected':
                self.model_selected.emit(data.get('data', {}).get('id', ''))
            elif msg_type == 'deselected':
                self.model_deselected.emit()
            elif msg_type == 'modelDeleted':
                self.model_deleted.emit(data.get('data', {}).get('id', ''))
            elif msg_type == 'modelTransformed':
                self.model_transformed.emit(data.get('data', {}))
            elif msg_type == 'requestAddModel':
                self.request_add_model.emit()
            elif msg_type == 'requestPointCloudOverlay':
                self.request_point_cloud_overlay.emit()
            else:
                self.message_received.emit(message)
                
        except json.JSONDecodeError as e:
            print(f"[ThreeJSBridge] JSON解析错误: {e}")
        except Exception as e:
            print(f"[ThreeJSBridge] 处理消息时出错: {e}")


class ThreeJSWebPage(QWebEnginePage):
    """自定义Web页面类"""
    
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        level_name = ['Info', 'Warning', 'Error'][level]
        print(f"[ThreeJS {level_name}] {message} (line {lineNumber})")


class ThreeJSSplatViewer(QWidget):
    """Three.js 3DGS分割查看器Widget"""
    
    # 信号
    viewer_ready = pyqtSignal()
    model_loaded = pyqtSignal(bool, str)
    split_complete = pyqtSignal(str)  # 分割后的模型路径
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.html_path = os.path.join(os.path.dirname(__file__), 'threejs_splat_viewer.html')
        self.current_model_path = None
        self.segmentation_mask = None
        self.selected_points_indices = None
        
        # 等待标志
        self._waiting_for_screenshot = False
        self._screenshot_result = None
        
        self.init_ui()
        self.load_viewer()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建Web视图
        self.web_view = QWebEngineView()
        
        # 配置WebEngine设置
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent, True)
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        
        # 使用自定义页面类
        page = ThreeJSWebPage(self.web_view)
        self.web_view.setPage(page)
        
        # 设置Web Channel
        self.channel = QWebChannel()
        self.bridge = ThreeJSBridge(self)
        self.channel.registerObject('pyBridge', self.bridge)
        page.setWebChannel(self.channel)
        
        # 连接信号
        self.bridge.viewer_ready.connect(self._on_viewer_ready)
        self.bridge.model_loaded.connect(self._on_model_loaded)
        self.bridge.screenshot_captured.connect(self._on_screenshot_captured)
        self.bridge.segmentation_complete.connect(self._on_segmentation_complete)
        self.bridge.points_selected.connect(self._on_points_selected)
        self.bridge.model_split_complete.connect(self._on_model_split_complete)
        
        layout.addWidget(self.web_view)
        
    def load_viewer(self):
        """加载Three.js查看器HTML（通过HTTP服务器以避免CORS问题）"""
        if not os.path.exists(self.html_path):
            # 如果HTML文件不存在，创建它
            self._create_viewer_html()
        
        # 通过HTTP服务器加载HTML，避免CORS问题
        try:
            from temp_http_server import get_global_server, add_file_to_server
            
            # 获取或启动HTTP服务器
            server = get_global_server()
            if not server.is_running():
                server.start()
            
            # 将HTML文件添加到HTTP服务器
            html_url = add_file_to_server(os.path.abspath(self.html_path), 'threejs_splat_viewer.html')
            
            # 同时添加依赖的库文件
            base_dir = os.path.dirname(self.html_path)
            lib_dir = os.path.join(base_dir, 'lib')
            
            if os.path.exists(lib_dir):
                # 添加Three.js
                threejs_path = os.path.join(lib_dir, 'threejs', 'three.min.js')
                if os.path.exists(threejs_path):
                    add_file_to_server(threejs_path, 'lib/threejs/three.min.js')
                    print(f"[ThreeJSSplatViewer] 添加Three.js: {threejs_path}")
                
                # 添加GaussianSplats3D
                gs3d_path = os.path.join(lib_dir, 'gaussian-splats-3d', 'gaussian-splats-3d.umd.cjs')
                if os.path.exists(gs3d_path):
                    add_file_to_server(gs3d_path, 'lib/gaussian-splats-3d/gaussian-splats-3d.umd.cjs')
                    print(f"[ThreeJSSplatViewer] 添加GaussianSplats3D: {gs3d_path}")
                
                # 添加Cesium相关文件（如果存在）
                cesium_dir = os.path.join(lib_dir, 'Cesium')
                if os.path.exists(cesium_dir):
                    # 只添加必要的文件
                    cesium_js = os.path.join(cesium_dir, 'Cesium.js')
                    if os.path.exists(cesium_js):
                        add_file_to_server(cesium_js, 'lib/Cesium/Cesium.js')
            
            print(f"[ThreeJSSplatViewer] 通过HTTP加载HTML: {html_url}")
            self.web_view.load(QUrl(html_url))
            
        except Exception as e:
            print(f"[ThreeJSSplatViewer] HTTP服务器不可用，使用file://协议: {e}")
            import traceback
            traceback.print_exc()
            url = QUrl.fromLocalFile(os.path.abspath(self.html_path))
            self.web_view.load(url)
        
    def _create_viewer_html(self):
        """创建Three.js查看器HTML文件"""
        # HTML内容将在单独的文件中创建
        pass
        
    def send_message(self, message: Dict[str, Any]):
        """发送消息到JavaScript"""
        message_json = json.dumps(message, ensure_ascii=False)
        message_json_escaped = message_json.replace("'", "\\'")
        js_code = f"if (typeof receiveMessageFromPython !== 'undefined') {{ receiveMessageFromPython('{message_json_escaped}'); }}"
        self.web_view.page().runJavaScript(js_code)
        
    def load_model(self, file_path: str, fly_to: bool = True):
        """
        加载3DGS模型文件
        
        Args:
            file_path: .ply或.splat文件路径
            fly_to: 是否自动调整视角
        """
        self.current_model_path = file_path
        
        # 需要通过HTTP服务器提供文件，因为JavaScript无法直接访问file://协议
        if file_path.startswith('http://') or file_path.startswith('https://'):
            file_url = file_path
        else:
            # 启动HTTP服务器并获取URL
            file_url = self._get_http_url_for_file(file_path)
            
        message = {
            'type': 'loadModel',
            'data': {
                'url': file_url,
                'flyTo': fly_to
            }
        }
        self.send_message(message)
    
    def _get_http_url_for_file(self, file_path: str) -> str:
        """
        为本地文件获取HTTP URL
        
        Args:
            file_path: 本地文件路径
            
        Returns:
            HTTP URL
        """
        try:
            from temp_http_server import get_global_server, add_file_to_server
            
            # 获取或启动HTTP服务器，并添加文件
            file_url = add_file_to_server(os.path.abspath(file_path))
            print(f"[ThreeJSSplatViewer] 文件HTTP URL: {file_url}")
            return file_url
                
        except ImportError:
            print("[ThreeJSSplatViewer] temp_http_server模块不可用，尝试使用file://协议")
            return f"file://{os.path.abspath(file_path)}"
        except Exception as e:
            print(f"[ThreeJSSplatViewer] 获取HTTP URL失败: {e}")
            import traceback
            traceback.print_exc()
            return f"file://{os.path.abspath(file_path)}"
        self.send_message(message)
        
    def capture_screenshot(self) -> Optional[QImage]:
        """捕获当前视角截图"""
        from PyQt5.QtCore import QByteArray
        
        self._waiting_for_screenshot = True
        self._screenshot_result = None
        
        message = {'type': 'captureScreenshot'}
        self.send_message(message)
        
        # 等待结果
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timeout_timer = QTimer()
        timeout_timer.setSingleShot(True)
        
        def check_result():
            if not self._waiting_for_screenshot:
                loop.quit()
            else:
                timer.start(50)
                
        def timeout():
            self._waiting_for_screenshot = False
            loop.quit()
            
        timer.timeout.connect(check_result)
        timeout_timer.timeout.connect(timeout)
        timeout_timer.start(5000)
        timer.start(50)
        loop.exec_()
        
        return self._screenshot_result
        
    def apply_segmentation_mask(self, mask: np.ndarray):
        """
        应用分割mask到模型
        
        Args:
            mask: 分割mask数组
        """
        self.segmentation_mask = mask
        
        # 将mask转换为可传输的格式
        mask_list = mask.flatten().tolist()
        
        message = {
            'type': 'applySegmentationMask',
            'data': {
                'mask': mask_list,
                'width': mask.shape[1] if len(mask.shape) > 1 else mask.shape[0],
                'height': mask.shape[0] if len(mask.shape) > 1 else 1
            }
        }
        self.send_message(message)
        
    def select_points_by_mask(self, keep_selected: bool = True):
        """
        根据分割mask选择高斯点
        
        Args:
            keep_selected: True保留选中的点，False删除选中的点
        """
        message = {
            'type': 'selectPointsByMask',
            'data': {
                'keepSelected': keep_selected
            }
        }
        self.send_message(message)
        
    def delete_unselected_points(self):
        """删除未选中的高斯点"""
        message = {'type': 'deleteUnselectedPoints'}
        self.send_message(message)
        
    def export_split_model(self, output_path: str = None) -> str:
        """
        导出分割后的模型
        
        Args:
            output_path: 输出路径，如果为None则自动生成
            
        Returns:
            导出的文件路径
        """
        if output_path is None and self.current_model_path:
            base, ext = os.path.splitext(self.current_model_path)
            output_path = f"{base}_split.splat"
            
        message = {
            'type': 'exportSplitModel',
            'data': {
                'outputPath': output_path
            }
        }
        self.send_message(message)
        
        return output_path
        
    def reset_view(self):
        """重置视角"""
        message = {'type': 'resetView'}
        self.send_message(message)
    
    def highlight_selected_points(self, indices: list, positions: list = None):
        """
        高亮显示选中的高斯点
        
        Args:
            indices: 选中点的索引列表
            positions: 可选的3D坐标列表 [[x,y,z], ...]
        """
        data = {
            'indices': indices,
            'count': len(indices)
        }
        
        # 如果提供了3D坐标，也发送过去
        if positions is not None:
            # 采样以减少数据量
            max_points = 50000
            if len(positions) > max_points:
                sample_rate = len(positions) // max_points
                sampled_positions = [positions[i] for i in range(0, len(positions), sample_rate)]
                data['positions'] = sampled_positions
                print(f"[ThreeJSSplatViewer] 采样3D坐标: {len(sampled_positions)} / {len(positions)}")
            else:
                data['positions'] = positions
        
        message = {
            'type': 'highlightSelectedPoints',
            'data': data
        }
        self.send_message(message)
        print(f"[ThreeJSSplatViewer] 发送高亮请求: {len(indices)} 个点")
        
    def clear_model(self):
        """清除当前模型"""
        message = {'type': 'clearModel'}
        self.send_message(message)
        self.current_model_path = None
        self.segmentation_mask = None
        self.selected_points_indices = None
    
    # ==================== 新增编辑功能 ====================
    
    def set_render_mode(self, mode: str):
        """设置渲染模式 ('splat', 'points', 'centers')"""
        message = {
            'type': 'setRenderMode',
            'data': {'mode': mode}
        }
        self.send_message(message)
    
    def send_point_cloud_data(self, positions: np.ndarray, colors: np.ndarray = None, 
                               selected: set = None, hidden: set = None, locked: set = None):
        """
        发送点云数据到JavaScript端用于渲染（向量化优化版）
        
        Args:
            positions: Nx3 numpy数组，点的3D坐标
            colors: Nx3 numpy数组，点的RGB颜色 (0-1范围)，可选
            selected: 选中点的索引集合
            hidden: 隐藏点的索引集合
            locked: 锁定点的索引集合
        """
        if positions is None or len(positions) == 0:
            return
        
        n = len(positions)
        selected = selected or set()
        hidden = hidden or set()
        locked = locked or set()
        
        # 创建索引数组
        indices = np.arange(n)
        
        # 创建隐藏mask
        hidden_mask = np.zeros(n, dtype=bool)
        if hidden:
            hidden_array = np.array(list(hidden), dtype=np.int32)
            hidden_array = hidden_array[hidden_array < n]  # 过滤越界索引
            hidden_mask[hidden_array] = True
        
        # 获取可见点索引
        visible_indices = indices[~hidden_mask]
        
        # 采样以减少数据量
        max_points = 300000
        if len(visible_indices) > max_points:
            sample_rate = len(visible_indices) // max_points
            visible_indices = visible_indices[::sample_rate]
        
        if len(visible_indices) == 0:
            return
        
        # 获取可见点的位置
        pos_array = positions[visible_indices].astype(np.float32)
        
        # 创建颜色数组 (RGBA)
        color_array = np.ones((len(visible_indices), 4), dtype=np.float32)
        
        # 如果有原始颜色，使用原始颜色
        if colors is not None and len(colors) > 0:
            valid_color_mask = visible_indices < len(colors)
            color_array[valid_color_mask, :3] = colors[visible_indices[valid_color_mask]]
        else:
            color_array[:, :3] = 0.7  # 没有颜色时使用默认灰色
        
        # 只有在有选中或锁定点时才覆盖颜色
        selected_mask = None
        
        # 设置选中点颜色（黄色）- 只覆盖选中的点
        if selected:
            selected_array = np.array(list(selected), dtype=np.int32)
            selected_mask = np.isin(visible_indices, selected_array)
            color_array[selected_mask] = [1.0, 1.0, 0.0, 1.0]
        
        # 设置锁定点颜色（橙色）- 只覆盖锁定但未选中的点
        if locked:
            locked_array = np.array(list(locked), dtype=np.int32)
            locked_mask = np.isin(visible_indices, locked_array)
            # 锁定但未选中的点显示橙色
            if selected_mask is not None:
                locked_not_selected = locked_mask & ~selected_mask
            else:
                locked_not_selected = locked_mask
            color_array[locked_not_selected] = [1.0, 0.5, 0.0, 1.0]
        
        # 展平数组并转换为列表
        pos_flat = pos_array.flatten().tolist()
        color_flat = color_array.flatten().tolist()
        
        message = {
            'type': 'createPointCloudFromData',
            'data': {
                'positions': pos_flat,
                'colors': color_flat
            }
        }
        self.send_message(message)
        print(f"[ThreeJSSplatViewer] 发送点云数据: {len(visible_indices)} 个点")
    
    def update_selection(self, selected: list, hidden: list = None, locked: list = None):
        """更新选择状态到JavaScript端"""
        message = {
            'type': 'updateSelection',
            'data': {
                'selected': selected,
                'hidden': hidden or [],
                'locked': locked or []
            }
        }
        self.send_message(message)
    
    def send_highlight_data(self, positions: list, bounding_box: dict = None):
        """
        发送高亮数据到JavaScript端
        
        Args:
            positions: 展平的坐标列表 [x1,y1,z1, x2,y2,z2, ...]
            bounding_box: 边界框 {'minX', 'minY', 'minZ', 'maxX', 'maxY', 'maxZ'}
        """
        message = {
            'type': 'updateHighlightFromData',
            'data': {
                'positions': positions or [],
                'boundingBox': bounding_box
            }
        }
        self.send_message(message)
    
    def send_point_cloud_overlay(self, positions: np.ndarray, colors: np.ndarray = None):
        """
        发送点云叠加数据到JavaScript端（用于在Splat渲染模式下叠加显示点云）
        
        Args:
            positions: Nx3 numpy数组，点的3D坐标
            colors: Nx3 numpy数组，点的RGB颜色 (0-1范围)，可选
        """
        if positions is None or len(positions) == 0:
            return
        
        # 采样以减少数据量
        max_points = 100000
        if len(positions) > max_points:
            sample_rate = len(positions) // max_points
            positions = positions[::sample_rate]
            if colors is not None:
                colors = colors[::sample_rate]
        
        pos_flat = positions.astype(np.float32).flatten().tolist()
        color_flat = colors.astype(np.float32).flatten().tolist() if colors is not None else []
        
        message = {
            'type': 'createPointCloudOverlay',
            'data': {
                'positions': pos_flat,
                'colors': color_flat
            }
        }
        self.send_message(message)
        print(f"[ThreeJSSplatViewer] 发送点云叠加数据: {len(positions)} 个点")
    
    def select_all_points(self):
        """全选所有点"""
        message = {'type': 'selectAll'}
        self.send_message(message)
    
    def select_inverse_points(self):
        """反选"""
        message = {'type': 'selectInverse'}
        self.send_message(message)
    
    def clear_selection(self):
        """清除选择"""
        message = {'type': 'clearSelection'}
        self.send_message(message)
        
    # 信号处理
    def _on_viewer_ready(self):
        print("[ThreeJSSplatViewer] Viewer已准备就绪")
        self.viewer_ready.emit()
        
    def _on_model_loaded(self, success: bool, error: str):
        if success:
            print("[ThreeJSSplatViewer] 模型加载成功")
        else:
            print(f"[ThreeJSSplatViewer] 模型加载失败: {error}")
        self.model_loaded.emit(success, error)
        
    def _on_screenshot_captured(self, data: dict):
        try:
            from PyQt5.QtCore import QByteArray
            data_dict = data.get('data', {})
            image_data_str = data_dict.get('imageData', '')
            
            if image_data_str and image_data_str.startswith('data:image'):
                base64_data = image_data_str.split(',')[1]
                image_data = QByteArray.fromBase64(base64_data.encode())
                image = QImage.fromData(image_data, 'PNG')
                if not image.isNull():
                    self._screenshot_result = image
            
            # 保存相机参数
            self._camera_position = data_dict.get('cameraPosition', None)
            self._camera_target = data_dict.get('cameraTarget', None)
            self._camera_fov = data_dict.get('cameraFov', 75)
            self._view_matrix = data_dict.get('viewMatrix', None)
            self._proj_matrix = data_dict.get('projMatrix', None)
            
            if self._camera_position:
                print(f"[ThreeJSSplatViewer] 相机位置: {self._camera_position}")
            if self._view_matrix:
                print(f"[ThreeJSSplatViewer] 已获取视图矩阵")
                
        except Exception as e:
            print(f"[ThreeJSSplatViewer] 处理截图数据失败: {e}")
        finally:
            self._waiting_for_screenshot = False
            
    def _on_segmentation_complete(self, data: dict):
        print(f"[ThreeJSSplatViewer] 分割完成: {data}")
        
    def _on_points_selected(self, data: dict):
        count = data.get('data', {}).get('selectedCount', 0)
        print(f"[ThreeJSSplatViewer] 已选择 {count} 个高斯点")
        
    def _on_model_split_complete(self, data: dict):
        output_path = data.get('data', {}).get('outputPath', '')
        success = data.get('success', False)
        if success:
            print(f"[ThreeJSSplatViewer] 模型分割完成: {output_path}")
            self.split_complete.emit(output_path)
        else:
            print(f"[ThreeJSSplatViewer] 模型分割失败")


class SplatSplitterPanel(QWidget):
    """3DGS分割操作面板"""
    
    # 信号
    load_to_cesium_requested = pyqtSignal(str, float, float, float, float)  # (path, lon, lat, alt, scale)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.viewer = None  # ThreeJSSplatViewer引用
        self.current_screenshot = None
        self.segmentation_mask = None
        self.sam3_api_url = "http://localhost:5000"
        self.current_model_path = None  # 当前加载的模型路径
        self.splitter = None  # GaussianSplatSplitter实例
        self.selected_indices = None  # 选中的高斯点索引
        self._is_previewing = False  # 是否正在预览分割部分
        self._preview_render_mode = 'splat'  # 预览渲染模式: 'splat' 或 'points'
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 模型加载组
        load_group = QGroupBox("模型加载")
        load_layout = QVBoxLayout()
        
        # 文件路径显示
        path_layout = QHBoxLayout()
        self.path_label = QLabel("未加载模型")
        self.path_label.setStyleSheet("color: #888; font-size: 11px;")
        path_layout.addWidget(self.path_label)
        load_layout.addLayout(path_layout)
        
        # 加载按钮
        self.load_btn = QPushButton("加载模型文件")
        self.load_btn.clicked.connect(self.on_load_model)
        load_layout.addWidget(self.load_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # 截图分割组
        segment_group = QGroupBox("截图分割")
        segment_layout = QVBoxLayout()
        
        # 截图按钮
        self.screenshot_btn = QPushButton("捕获截图")
        self.screenshot_btn.clicked.connect(self.on_capture_screenshot)
        self.screenshot_btn.setEnabled(False)
        segment_layout.addWidget(self.screenshot_btn)
        
        # 截图预览
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(280, 180)
        self.preview_label.setStyleSheet("background-color: #1E1E1E; border: 1px solid #444;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("等待截图...")
        segment_layout.addWidget(self.preview_label)
        
        # 分割提示词
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("提示词:"))
        self.prompt_input = QLineEdit("building")
        self.prompt_input.setPlaceholderText("如: building, car, tree")
        prompt_layout.addWidget(self.prompt_input)
        segment_layout.addLayout(prompt_layout)
        
        # SAM3分割按钮
        self.segment_btn = QPushButton("SAM3分割")
        self.segment_btn.clicked.connect(self.on_segment)
        self.segment_btn.setEnabled(False)
        self.segment_btn.setStyleSheet("background-color: #6C5CE7; color: white;")
        segment_layout.addWidget(self.segment_btn)
        
        # 分割状态
        self.segment_status = QLabel("")
        self.segment_status.setStyleSheet("color: #888; font-size: 11px;")
        segment_layout.addWidget(self.segment_status)
        
        segment_group.setLayout(segment_layout)
        layout.addWidget(segment_group)
        
        # 高斯点选择组
        select_group = QGroupBox("高斯点选择")
        select_layout = QVBoxLayout()
        
        # 选择模式
        self.keep_selected_checkbox = QCheckBox("保留选中区域（否则删除选中区域）")
        self.keep_selected_checkbox.setChecked(True)
        select_layout.addWidget(self.keep_selected_checkbox)
        
        # 应用分割按钮
        self.apply_mask_btn = QPushButton("应用分割到模型")
        self.apply_mask_btn.clicked.connect(self.on_apply_mask)
        self.apply_mask_btn.setEnabled(False)
        select_layout.addWidget(self.apply_mask_btn)
        
        # 预览分割部分按钮
        self.preview_split_btn = QPushButton("预览分割部分")
        self.preview_split_btn.clicked.connect(self.on_preview_split)
        self.preview_split_btn.setEnabled(False)
        self.preview_split_btn.setStyleSheet("background-color: #6C5CE7; color: white;")
        select_layout.addWidget(self.preview_split_btn)
        
        # 渲染模式切换（仅在预览时可用）
        render_mode_layout = QHBoxLayout()
        render_mode_layout.addWidget(QLabel("预览模式:"))
        self.preview_render_mode = QComboBox()
        self.preview_render_mode.addItems(["Splat渲染", "点云渲染"])
        self.preview_render_mode.currentIndexChanged.connect(self.on_preview_render_mode_changed)
        self.preview_render_mode.setEnabled(False)
        render_mode_layout.addWidget(self.preview_render_mode)
        select_layout.addLayout(render_mode_layout)
        
        # 删除未选中点按钮
        self.delete_btn = QPushButton("删除未选中的高斯点")
        self.delete_btn.clicked.connect(self.on_delete_unselected)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("background-color: #D63031; color: white;")
        select_layout.addWidget(self.delete_btn)
        
        select_group.setLayout(select_layout)
        layout.addWidget(select_group)
        
        # 导出到Cesium组
        export_group = QGroupBox("导出到Cesium")
        export_layout = QVBoxLayout()
        
        # 地理坐标设置
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("经度:"))
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180.0, 180.0)
        self.lon_input.setDecimals(6)
        self.lon_input.setValue(114.610945)
        coord_layout.addWidget(self.lon_input)
        export_layout.addLayout(coord_layout)
        
        coord_layout2 = QHBoxLayout()
        coord_layout2.addWidget(QLabel("纬度:"))
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90.0, 90.0)
        self.lat_input.setDecimals(6)
        self.lat_input.setValue(30.457906)
        coord_layout2.addWidget(self.lat_input)
        export_layout.addLayout(coord_layout2)
        
        coord_layout3 = QHBoxLayout()
        coord_layout3.addWidget(QLabel("高度:"))
        self.alt_input = QDoubleSpinBox()
        self.alt_input.setRange(-1000.0, 10000.0)
        self.alt_input.setDecimals(2)
        self.alt_input.setValue(0.0)
        self.alt_input.setSuffix(" m")
        coord_layout3.addWidget(self.alt_input)
        export_layout.addLayout(coord_layout3)
        
        coord_layout4 = QHBoxLayout()
        coord_layout4.addWidget(QLabel("缩放:"))
        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(0.001, 1000.0)
        self.scale_input.setDecimals(3)
        self.scale_input.setValue(1.0)
        coord_layout4.addWidget(self.scale_input)
        export_layout.addLayout(coord_layout4)
        
        # 导出并加载按钮
        self.export_btn = QPushButton("导出分割模型并加载到Cesium")
        self.export_btn.clicked.connect(self.on_export_to_cesium)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("background-color: #0984E3; color: white;")
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #5BA3D8; padding: 5px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
    def set_viewer(self, viewer: ThreeJSSplatViewer):
        """设置关联的查看器"""
        self.viewer = viewer
        self.viewer.model_loaded.connect(self._on_model_loaded)
        self.viewer.split_complete.connect(self._on_split_complete)
        
    def on_load_model(self):
        """加载模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择3DGS模型文件", "", 
            "Gaussian Splat Files (*.ply *.splat *.ksplat);;All Files (*)"
        )
        
        if file_path and self.viewer:
            self.current_model_path = file_path
            self.path_label.setText(os.path.basename(file_path))
            self.status_label.setText("正在加载模型...")
            
            # 如果是PLY文件，同时加载到Python端的分割器
            if file_path.lower().endswith('.ply'):
                self.splitter = GaussianSplatSplitter()
                if self.splitter.load_ply(file_path):
                    # 检查是否是压缩格式且未能解码
                    if getattr(self.splitter, '_is_compressed', False):
                        self.status_label.setText(f"⚠ 压缩格式PLY ({self.splitter.point_count} 点) - 坐标解码失败")
                        self.path_label.setStyleSheet("color: #FFD93D; font-size: 11px;")
                        QMessageBox.warning(
                            self, "压缩格式提示",
                            f"检测到压缩格式PLY文件：\n{os.path.basename(file_path)}\n\n"
                            "坐标数据解码失败，分割功能将受限。\n\n"
                            "如需完整的分割功能，请使用原始的 point_cloud.ply 文件，\n"
                            "通常位于 output/iteration_xxx/ 文件夹中。\n\n"
                            "当前仍可进行可视化预览。"
                        )
                    else:
                        # 成功解码（包括压缩格式成功解码的情况）
                        xyz = self.splitter.xyz
                        x_range = xyz[:, 0].max() - xyz[:, 0].min()
                        y_range = xyz[:, 1].max() - xyz[:, 1].min()
                        z_range = xyz[:, 2].max() - xyz[:, 2].min()
                        self.status_label.setText(f"正在加载模型... ({self.splitter.point_count} 个高斯点)")
                        self.path_label.setStyleSheet("color: #51CF66; font-size: 11px;")
                        print(f"[SplatSplitterPanel] 坐标范围: X={x_range:.2f}, Y={y_range:.2f}, Z={z_range:.2f}")
                else:
                    self.splitter = None
            
            self.viewer.load_model(file_path)
            
    def on_capture_screenshot(self):
        """捕获截图"""
        if not self.viewer:
            return
            
        self.status_label.setText("正在捕获截图...")
        self.current_screenshot = self.viewer.capture_screenshot()
        
        if self.current_screenshot:
            # 显示预览
            pixmap = QPixmap.fromImage(self.current_screenshot)
            scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            self.segment_btn.setEnabled(True)
            self.status_label.setText("截图完成，可以进行分割")
        else:
            self.status_label.setText("截图失败")
            
    def on_segment(self):
        """执行SAM3分割"""
        if not self.current_screenshot:
            return
            
        prompt = self.prompt_input.text().strip() or "building"
        self.segment_btn.setEnabled(False)
        self.segment_status.setText("正在分割...")
        self.status_label.setText("正在调用SAM3 API...")
        
        # 在后台线程执行分割
        from PyQt5.QtCore import QThread
        from gs_screenshot_splitter import call_sam3_api
        
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
                
        self.segment_thread = SegmentThread(self.current_screenshot, prompt, self.sam3_api_url)
        self.segment_thread.finished.connect(self._on_segment_finished)
        self.segment_thread.start()
        
    def _on_segment_finished(self, result: dict):
        """分割完成回调"""
        self.segment_btn.setEnabled(True)
        
        if result.get('success', False):
            geojson = result.get('geojson', {})
            summary = result.get('summary', '')
            
            if geojson and 'features' in geojson:
                # 从GeoJSON创建mask
                self.segmentation_mask = self._geojson_to_mask(
                    geojson,
                    self.current_screenshot.width(),
                    self.current_screenshot.height()
                )
                
                num_objects = len(geojson.get('features', []))
                self.segment_status.setText(f"✓ 检测到{num_objects}个对象")
                self.segment_status.setStyleSheet("color: #51CF66; font-size: 11px;")
                self.status_label.setText("分割完成，可以应用到模型")
                
                # 更新预览显示分割结果
                self._update_preview_with_mask()
                
                # 发送轮廓数据到JavaScript进行3D可视化
                self._send_contours_to_viewer(geojson)
                
                self.apply_mask_btn.setEnabled(True)
            else:
                self.segment_status.setText("未检测到对象")
                self.segment_status.setStyleSheet("color: #FFD93D; font-size: 11px;")
        else:
            error_msg = result.get('summary', '分割失败')
            self.segment_status.setText(f"✗ {error_msg}")
            self.segment_status.setStyleSheet("color: #FF6B6B; font-size: 11px;")
            self.status_label.setText("分割失败")
    
    def _send_contours_to_viewer(self, geojson: dict):
        """发送轮廓数据到JavaScript进行3D可视化"""
        if not self.viewer or not geojson:
            return
        
        try:
            contours = []
            
            for feature in geojson.get('features', []):
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'Polygon':
                    coordinates = geometry.get('coordinates', [])
                    if coordinates and len(coordinates) > 0:
                        ring = coordinates[0]
                        # 转换为简单的点列表 [[x, y], [x, y], ...]
                        contour = [[int(p[0]), int(p[1])] for p in ring]
                        contours.append(contour)
            
            if contours:
                self.viewer.send_message({
                    'type': 'visualizeContours',
                    'data': {
                        'contours': contours,
                        'imageWidth': self.current_screenshot.width(),
                        'imageHeight': self.current_screenshot.height()
                    }
                })
                print(f"[SplatSplitterPanel] 发送 {len(contours)} 个轮廓到3D视图")
        except Exception as e:
            print(f"[SplatSplitterPanel] 发送轮廓数据失败: {e}")
            
    def _geojson_to_mask(self, geojson: dict, width: int, height: int) -> np.ndarray:
        """将GeoJSON转换为mask"""
        import cv2
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if 'features' not in geojson:
            return mask
            
        try:
            for idx, feature in enumerate(geojson['features']):
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'Polygon':
                    coordinates = geometry.get('coordinates', [])
                    if coordinates and len(coordinates) > 0:
                        ring = coordinates[0]
                        points = np.array([[int(p[0]), int(p[1])] for p in ring], dtype=np.int32)
                        cv2.fillPoly(mask, [points], idx + 1)
        except Exception as e:
            print(f"转换GeoJSON到mask失败: {e}")
            
        return mask
        
    def _update_preview_with_mask(self):
        """更新预览显示分割结果"""
        if self.current_screenshot is None or self.segmentation_mask is None:
            return
            
        import cv2
        from PyQt5.QtGui import QPainter, QColor, QBrush, QPolygon
        
        # 创建带mask的预览图
        pixmap = QPixmap.fromImage(self.current_screenshot)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制mask
        mask_resized = cv2.resize(
            self.segmentation_mask.astype(np.uint8),
            (pixmap.width(), pixmap.height()),
            interpolation=cv2.INTER_NEAREST
        )
        
        unique_labels = np.unique(mask_resized)
        colors = [
            QColor(255, 0, 0, 100),
            QColor(0, 255, 0, 100),
            QColor(0, 0, 255, 100),
            QColor(255, 255, 0, 100),
        ]
        
        for idx, label in enumerate(unique_labels):
            if label == 0:
                continue
                
            color = colors[idx % len(colors)]
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            
            mask_label = (mask_resized == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                from PyQt5.QtCore import QPoint
                points = [QPoint(int(p[0][0]), int(p[0][1])) for p in contour]
                polygon = QPolygon(points)
                painter.drawPolygon(polygon)
                
        painter.end()
        
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        
    def on_apply_mask(self):
        """应用分割mask到模型 - 高亮选中的高斯点"""
        if self.segmentation_mask is None:
            QMessageBox.warning(self, "警告", "没有分割结果")
            return
        
        # 检查是否是压缩格式且未能解码
        if self.splitter and getattr(self.splitter, '_is_compressed', False):
            QMessageBox.warning(
                self, "压缩格式警告",
                "当前加载的压缩格式PLY文件无法完全解码坐标数据。\n\n"
                "可能的原因：\n"
                "1. 缺少chunk边界框信息\n"
                "2. 使用了不支持的压缩格式\n\n"
                "建议使用原始的 point_cloud.ply 文件，\n"
                "通常位于 output/iteration_xxx/ 文件夹中。"
            )
            return
            
        self.status_label.setText("正在应用分割到模型...")
        
        keep_selected = self.keep_selected_checkbox.isChecked()
        
        # 如果有Python端的分割器，使用Python端进行点选择
        if self.splitter and self.splitter.point_count > 0:
            # 检查坐标是否有效
            xyz = self.splitter.xyz
            x_range = xyz[:, 0].max() - xyz[:, 0].min()
            y_range = xyz[:, 1].max() - xyz[:, 1].min()
            z_range = xyz[:, 2].max() - xyz[:, 2].min()
            
            if x_range < 0.001 and y_range < 0.001 and z_range < 0.001:
                QMessageBox.warning(
                    self, "坐标数据无效",
                    "点云坐标数据无效（范围太小），可能是压缩格式文件。\n\n"
                    "请使用原始的 point_cloud.ply 文件进行分割。"
                )
                return
            
            self.status_label.setText("正在选择高斯点...")
            
            # 简化的选择方法：遍历所有点，检查是否在mask区域内
            self.selected_indices = self._select_points_simple(keep_selected)
            
            if self.selected_indices is not None and len(self.selected_indices) > 0:
                self.status_label.setText(f"已选择 {len(self.selected_indices)} / {self.splitter.point_count} 个高斯点")
                self.preview_split_btn.setEnabled(True)
                self.delete_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                
                # 通知JavaScript端高亮选中的点，同时传递3D坐标
                if self.viewer and self.splitter.xyz is not None:
                    selected_xyz = self.splitter.xyz[self.selected_indices]
                    # 转换为列表格式 [[x,y,z], ...]
                    positions_list = selected_xyz.tolist()
                    self.viewer.highlight_selected_points(
                        self.selected_indices.tolist(),
                        positions_list
                    )
            else:
                self.status_label.setText("未选择到任何点，请检查分割区域")
        else:
            # 使用JavaScript端的选择
            if self.viewer:
                self.viewer.apply_segmentation_mask(self.segmentation_mask)
                self.viewer.select_points_by_mask(keep_selected)
            self.preview_split_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
            self.status_label.setText("分割已应用，可以预览或删除")
    
    def on_preview_split(self):
        """预览分割部分 - 切换显示模式"""
        if not self.viewer:
            return
        
        self._is_previewing = not self._is_previewing
        
        if self._is_previewing:
            # 进入预览模式：只显示选中的部分
            if self.selected_indices is not None and len(self.selected_indices) > 0:
                if self.splitter and self.splitter.xyz is not None:
                    # 获取选中点的数据
                    selected_xyz = self.splitter.xyz[self.selected_indices]
                    selected_colors = None
                    
                    # 如果有颜色数据，也提取出来
                    if hasattr(self.splitter, 'colors') and self.splitter.colors is not None:
                        selected_colors = self.splitter.colors[self.selected_indices]
                        print(f"[SplatSplitterPanel] 提取颜色数据: 形状={selected_colors.shape}, dtype={selected_colors.dtype}")
                    else:
                        print(f"[SplatSplitterPanel] 警告: splitter没有颜色数据")
                    
                    # 隐藏原始模型
                    self.viewer.send_message({'type': 'hideOriginalModel'})
                    
                    # 准备位置和颜色数据
                    positions_flat = selected_xyz.astype(np.float32).flatten().tolist()
                    colors_flat = []
                    
                    if selected_colors is not None:
                        # 确保颜色在0-1范围内
                        if selected_colors.max() > 1.0:
                            # 如果是0-255范围，转换为0-1
                            colors_normalized = selected_colors.astype(np.float32) / 255.0
                        else:
                            colors_normalized = selected_colors.astype(np.float32)
                        colors_flat = colors_normalized.flatten().tolist()
                        print(f"[SplatSplitterPanel] 颜色数据: 形状={selected_colors.shape}, dtype={selected_colors.dtype}, 范围=[{selected_colors.min()}, {selected_colors.max()}]")
                        print(f"[SplatSplitterPanel] 归一化后: 形状={colors_normalized.shape}, 范围=[{colors_normalized.min():.3f}, {colors_normalized.max():.3f}]")
                        print(f"[SplatSplitterPanel] 前3个点的颜色: {colors_normalized[:3].tolist()}")
                        print(f"[SplatSplitterPanel] 发送数据长度: positions={len(positions_flat)}, colors={len(colors_flat)}")
                    else:
                        print(f"[SplatSplitterPanel] 警告: 没有颜色数据")
                    
                    # 根据预览模式渲染
                    if self._preview_render_mode == 'splat':
                        # Splat渲染模式：导出临时splat文件并加载
                        self.status_label.setText("正在生成预览splat文件...")
                        
                        # 生成临时splat文件
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_splat_path = os.path.join(temp_dir, 'preview_temp.splat')
                        
                        splat_url = None
                        if self.splitter.export_selected_to_splat(self.selected_indices, temp_splat_path):
                            # 通过HTTP服务器提供文件
                            try:
                                from temp_http_server import add_file_to_server
                                splat_url = add_file_to_server(temp_splat_path, 'preview_temp.splat')
                                print(f"[SplatSplitterPanel] Splat文件已准备: {splat_url}")
                            except Exception as e:
                                print(f"[SplatSplitterPanel] 提供splat文件失败: {e}")
                        
                        # 发送splat URL
                        self.viewer.send_message({
                            'type': 'previewSplitAsSplat',
                            'data': {
                                'splatUrl': splat_url
                            }
                        })
                        
                        if splat_url:
                            self.status_label.setText(f"预览模式：Splat渲染 ({len(self.selected_indices)} 个点)")
                        else:
                            self.status_label.setText("Splat预览失败：无法生成临时文件")
                    else:
                        # 点云渲染模式：发送点云数据
                        self.viewer.send_message({
                            'type': 'previewSplitAsPoints',
                            'data': {
                                'positions': positions_flat,
                                'colors': colors_flat
                            }
                        })
                        
                        self.status_label.setText(f"预览模式：点云渲染 ({len(self.selected_indices)} 个点)")
                    
                    self.preview_split_btn.setText("显示原模型")
                    self.preview_split_btn.setStyleSheet("background-color: #00B894; color: white;")
                    self.preview_render_mode.setEnabled(True)
                    self.status_label.setText(f"预览模式：显示 {len(self.selected_indices)} 个点")
            else:
                self.status_label.setText("没有选中的点可预览")
                self._is_previewing = False
        else:
            # 退出预览模式：显示原模型
            self.viewer.send_message({'type': 'showOriginalModel'})
            self.viewer.send_message({'type': 'clearPreview'})
            
            self.preview_split_btn.setText("预览分割部分")
            self.preview_split_btn.setStyleSheet("background-color: #6C5CE7; color: white;")
            self.preview_render_mode.setEnabled(False)
            self.status_label.setText("已恢复原模型显示")
    
    def on_preview_render_mode_changed(self, index):
        """预览渲染模式切换"""
        if not self._is_previewing:
            return
        
        self._preview_render_mode = 'splat' if index == 0 else 'points'
        
        # 重新触发预览以应用新的渲染模式
        self._is_previewing = False  # 临时设为False
        self.on_preview_split()  # 重新进入预览模式
    
    def _select_points_simple(self, keep_inside: bool) -> np.ndarray:
        """
        基于mask的点选择方法 - 将3D点投影到2D并检查是否在mask内
        
        使用与Three.js相机一致的透视投影。
        需要从JavaScript端获取相机参数，或者使用简化的投影方法。
        
        Args:
            keep_inside: True保留mask内的点，False保留mask外的点
            
        Returns:
            选中点的索引数组
        """
        if self.splitter is None or self.segmentation_mask is None:
            print("[SplatSplitterPanel] 无法选择点：splitter或mask为空")
            return np.array([], dtype=np.int32)
        
        # 检查是否是压缩格式且未能解码
        if getattr(self.splitter, '_is_compressed', False):
            print("[SplatSplitterPanel] 警告：压缩格式PLY文件坐标解码失败，无法进行精确的点选择")
            print("[SplatSplitterPanel] 建议使用原始的point_cloud.ply文件（通常在iteration_xxx文件夹中）")
            return np.array([], dtype=np.int32)
        
        try:
            xyz = self.splitter.xyz
            mask = self.segmentation_mask
            mask_height, mask_width = mask.shape
            
            print(f"[SplatSplitterPanel] 开始选择点: {self.splitter.point_count} 个点, mask尺寸: {mask_width}x{mask_height}")
            
            # 获取高斯点的边界
            x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
            y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
            z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
            
            print(f"[SplatSplitterPanel] 点云边界: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
            
            # 计算范围
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            # 检查坐标是否有效
            if max(x_range, y_range, z_range) < 0.001:
                print("[SplatSplitterPanel] 警告：点云坐标范围太小，可能是数据问题")
                return np.array([], dtype=np.int32)
            
            # 尝试从viewer获取相机参数
            camera_pos = None
            camera_target = None
            fov = 75
            view_matrix = None
            proj_matrix = None
            
            if self.viewer:
                camera_pos = getattr(self.viewer, '_camera_position', None)
                camera_target = getattr(self.viewer, '_camera_target', None)
                fov = getattr(self.viewer, '_camera_fov', 75)
                view_matrix = getattr(self.viewer, '_view_matrix', None)
                proj_matrix = getattr(self.viewer, '_proj_matrix', None)
            
            # 如果有完整的矩阵，使用矩阵投影
            if view_matrix is not None and proj_matrix is not None:
                print("[SplatSplitterPanel] 使用JavaScript传来的相机矩阵进行投影")
                selected = self._select_points_with_matrices(
                    xyz, mask, view_matrix, proj_matrix, 
                    mask_width, mask_height, keep_inside
                )
                if selected is not None and len(selected) > 0:
                    return selected
                print("[SplatSplitterPanel] 矩阵投影失败，回退到简化投影")
            
            # 否则使用相机位置和目标进行简化投影
            if camera_pos is not None:
                camera_pos = np.array(camera_pos)
                print(f"[SplatSplitterPanel] 使用相机位置: {camera_pos}")
            else:
                # 默认相机位置
                camera_pos = np.array([0, 5, 15])
                print("[SplatSplitterPanel] 使用默认相机位置")
            
            if camera_target is not None:
                camera_target = np.array(camera_target)
            else:
                # 默认朝向点云中心
                camera_target = np.array([
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    (z_min + z_max) / 2
                ])
            
            # 计算相机坐标系
            forward = camera_target - camera_pos
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            
            # 假设up向量为Y轴
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                # forward与up平行，使用Z轴作为参考
                up = np.array([0, 0, 1])
                right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)
            
            # 将所有点转换到相机坐标系
            points_centered = xyz - camera_pos
            
            # 投影到相机坐标系
            cam_x = np.dot(points_centered, right)
            cam_y = np.dot(points_centered, up)
            cam_z = np.dot(points_centered, forward)  # 深度（正值表示在相机前方）
            
            # 透视投影 - 只处理在相机前方的点
            valid_depth = cam_z > 0.1
            
            # FOV计算投影
            fov_rad = np.radians(fov)
            aspect = mask_width / mask_height
            
            # 透视除法
            proj_x = np.zeros_like(cam_x)
            proj_y = np.zeros_like(cam_y)
            
            proj_x[valid_depth] = cam_x[valid_depth] / (cam_z[valid_depth] * np.tan(fov_rad / 2) * aspect)
            proj_y[valid_depth] = cam_y[valid_depth] / (cam_z[valid_depth] * np.tan(fov_rad / 2))
            
            # 转换到屏幕坐标 [0, width] 和 [0, height]
            screen_x = ((proj_x + 1) / 2 * mask_width).astype(np.int32)
            screen_y = ((1 - proj_y) / 2 * mask_height).astype(np.int32)  # Y轴翻转
            
            # 裁剪到有效范围
            valid_x = (screen_x >= 0) & (screen_x < mask_width)
            valid_y = (screen_y >= 0) & (screen_y < mask_height)
            valid = valid_depth & valid_x & valid_y
            
            # 获取有效点的索引和屏幕坐标
            valid_indices = np.where(valid)[0]
            sx = screen_x[valid_indices]
            sy = screen_y[valid_indices]
            
            # 批量获取mask值（向量化操作，避免循环）
            mask_values = mask[sy, sx]
            
            # 根据mask值选择点
            if keep_inside:
                selected_local = mask_values > 0
            else:
                selected_local = mask_values == 0
            
            result = valid_indices[selected_local].astype(np.int32)
            print(f"[SplatSplitterPanel] 简化投影: {len(result)}/{self.splitter.point_count} 点 (有效:{len(valid_indices)})")
            return result
            
        except Exception as e:
            print(f"[SplatSplitterPanel] 点选择失败: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int32)
    
    def _select_points_with_matrices(self, xyz: np.ndarray, mask: np.ndarray,
                                      view_matrix: list, proj_matrix: list,
                                      mask_width: int, mask_height: int,
                                      keep_inside: bool) -> np.ndarray:
        """
        使用相机矩阵进行精确的3D到2D投影，选择mask区域内的点
        
        优化版本：使用float32、向量化操作、减少内存分配
        
        Args:
            xyz: 点云坐标 (N, 3)
            mask: 分割mask (height, width)
            view_matrix: 视图矩阵（16个元素，列主序）
            proj_matrix: 投影矩阵（16个元素，列主序）
            mask_width: mask宽度
            mask_height: mask高度
            keep_inside: True保留mask内的点，False保留mask外的点
            
        Returns:
            选中点的索引数组
        """
        try:
            n_points = len(xyz)
            
            # 将列表转换为4x4矩阵（Three.js使用列主序，转置为行主序）
            # Three.js elements 是按列存储的: [m00,m10,m20,m30, m01,m11,m21,m31, ...]
            # 所以 reshape(4,4) 得到的是转置矩阵，再 .T 得到正确的行主序矩阵
            view_mat = np.array(view_matrix, dtype=np.float32).reshape(4, 4).T
            proj_mat = np.array(proj_matrix, dtype=np.float32).reshape(4, 4).T
            
            # 调试：打印矩阵的关键元素
            print(f"[SplatSplitterPanel] 视图矩阵对角线: {view_mat[0,0]:.3f}, {view_mat[1,1]:.3f}, {view_mat[2,2]:.3f}")
            print(f"[SplatSplitterPanel] 视图矩阵平移: {view_mat[0,3]:.3f}, {view_mat[1,3]:.3f}, {view_mat[2,3]:.3f}")
            print(f"[SplatSplitterPanel] 投影矩阵: fov相关={proj_mat[0,0]:.3f}, {proj_mat[1,1]:.3f}")
            
            # 组合MVP矩阵（一次矩阵乘法）
            mvp = np.ascontiguousarray(proj_mat @ view_mat)
            
            # 确保xyz是float32且连续内存
            if xyz.dtype != np.float32:
                xyz_f32 = xyz.astype(np.float32)
            else:
                xyz_f32 = np.ascontiguousarray(xyz)
            
            # 高效的齐次坐标变换：直接计算clip坐标，避免创建大的中间数组
            # clip = mvp @ [x, y, z, 1]^T
            # 分解为: clip_x = m00*x + m01*y + m02*z + m03
            #         clip_y = m10*x + m11*y + m12*z + m13
            #         clip_z = m20*x + m21*y + m22*z + m23
            #         clip_w = m30*x + m31*y + m32*z + m33
            
            x, y, z = xyz_f32[:, 0], xyz_f32[:, 1], xyz_f32[:, 2]
            
            clip_w = mvp[3, 0] * x + mvp[3, 1] * y + mvp[3, 2] * z + mvp[3, 3]
            
            # 只处理w > 0的点（在相机前方）
            valid_w = clip_w > 0.001
            valid_count = np.sum(valid_w)
            
            if valid_count == 0:
                print(f"[SplatSplitterPanel] 没有点在相机前方")
                return np.array([], dtype=np.int32)
            
            # 只对有效点计算完整的投影（节省计算）
            valid_idx = np.where(valid_w)[0]
            x_v, y_v, z_v = x[valid_w], y[valid_w], z[valid_w]
            w_v = clip_w[valid_w]
            
            # 计算NDC坐标（透视除法）
            inv_w = 1.0 / w_v
            ndc_x = (mvp[0, 0] * x_v + mvp[0, 1] * y_v + mvp[0, 2] * z_v + mvp[0, 3]) * inv_w
            ndc_y = (mvp[1, 0] * x_v + mvp[1, 1] * y_v + mvp[1, 2] * z_v + mvp[1, 3]) * inv_w
            
            # 调试：打印NDC坐标范围
            print(f"[SplatSplitterPanel] NDC范围: X[{ndc_x.min():.2f}, {ndc_x.max():.2f}], Y[{ndc_y.min():.2f}, {ndc_y.max():.2f}]")
            
            # 检查是否在视锥体内 (-1 到 1)，只检查x和y
            in_frustum = (ndc_x >= -1) & (ndc_x <= 1) & (ndc_y >= -1) & (ndc_y <= 1)
            
            # 转换为屏幕坐标
            screen_x = ((ndc_x + 1) * 0.5 * mask_width).astype(np.int32)
            screen_y = ((1 - ndc_y) * 0.5 * mask_height).astype(np.int32)
            
            # 裁剪到有效范围
            valid_screen = in_frustum & \
                           (screen_x >= 0) & (screen_x < mask_width) & \
                           (screen_y >= 0) & (screen_y < mask_height)
            
            # 获取在屏幕范围内的点的原始索引
            screen_valid_idx = valid_idx[valid_screen]
            sx = screen_x[valid_screen]
            sy = screen_y[valid_screen]
            
            # 批量获取mask值
            mask_values = mask[sy, sx]
            
            # 根据mask值选择点
            if keep_inside:
                selected_local = mask_values > 0
            else:
                selected_local = mask_values == 0
            
            result = screen_valid_idx[selected_local].astype(np.int32)
            
            print(f"[SplatSplitterPanel] 矩阵投影: {len(result)}/{n_points} 点 (视锥内:{valid_count}, 屏幕内:{len(sx)})")
            
            return result
            
        except Exception as e:
            print(f"[SplatSplitterPanel] 矩阵投影失败: {e}")
            import traceback
            traceback.print_exc()
            return np.array([], dtype=np.int32)
        
    def on_delete_unselected(self):
        """删除未选中的高斯点 - 真正删除数据"""
        # 如果还没有选择点，先尝试选择
        if self.selected_indices is None or len(self.selected_indices) == 0:
            if self.segmentation_mask is not None and self.splitter is not None:
                # 自动执行点选择
                keep_selected = self.keep_selected_checkbox.isChecked()
                self.selected_indices = self._select_points_simple(keep_selected)
            
        if self.selected_indices is None or len(self.selected_indices) == 0:
            QMessageBox.warning(self, "警告", "没有选中任何点，请先进行分割并应用")
            return
            
        total_points = self.splitter.point_count if self.splitter else 0
        delete_count = total_points - len(self.selected_indices)
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"将永久删除 {delete_count} 个高斯点，保留 {len(self.selected_indices)} 个点。\n"
            f"此操作不可撤销，确定继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 在Python端真正删除数据
            if self.splitter:
                # 更新splitter的数据，只保留选中的点
                self.splitter.xyz = self.splitter.xyz[self.selected_indices]
                if hasattr(self.splitter, 'colors') and self.splitter.colors is not None:
                    self.splitter.colors = self.splitter.colors[self.selected_indices]
                if hasattr(self.splitter, 'scales') and self.splitter.scales is not None:
                    self.splitter.scales = self.splitter.scales[self.selected_indices]
                if hasattr(self.splitter, 'rotations') and self.splitter.rotations is not None:
                    self.splitter.rotations = self.splitter.rotations[self.selected_indices]
                if hasattr(self.splitter, 'opacities') and self.splitter.opacities is not None:
                    self.splitter.opacities = self.splitter.opacities[self.selected_indices]
                if hasattr(self.splitter, 'features_dc') and self.splitter.features_dc is not None:
                    self.splitter.features_dc = self.splitter.features_dc[self.selected_indices]
                if hasattr(self.splitter, 'features_rest') and self.splitter.features_rest is not None:
                    self.splitter.features_rest = self.splitter.features_rest[self.selected_indices]
                
                self.splitter.point_count = len(self.selected_indices)
                
                # 重置选中索引为全部（因为现在只有这些点了）
                self.selected_indices = np.arange(self.splitter.point_count, dtype=np.int32)
            
            # 在JavaScript端也执行删除
            if self.viewer:
                # 清除预览
                if self._is_previewing:
                    self._is_previewing = False
                    self.preview_split_btn.setText("预览分割部分")
                    self.preview_split_btn.setStyleSheet("background-color: #6C5CE7; color: white;")
                    self.preview_render_mode.setEnabled(False)
                
                # 发送删除命令，只保留选中的点
                self.viewer.send_message({
                    'type': 'deleteUnselectedPointsPermanently',
                    'data': {
                        'selectedIndices': self.selected_indices.tolist()
                    }
                })
            
            self.status_label.setText(f"已删除 {delete_count} 个点，当前模型有 {len(self.selected_indices)} 个点")
            self.export_btn.setEnabled(True)
            self.delete_btn.setEnabled(False)  # 删除后禁用删除按钮
            self.preview_split_btn.setEnabled(False)  # 删除后禁用预览按钮
            
    def on_export_to_cesium(self):
        """导出分割模型并加载到Cesium"""
        if not self.current_model_path:
            QMessageBox.warning(self, "警告", "没有加载模型")
            return
        
        # 生成输出路径
        base, ext = os.path.splitext(self.current_model_path)
        output_splat_path = f"{base}_split.splat"
        
        lon = self.lon_input.value()
        lat = self.lat_input.value()
        alt = self.alt_input.value()
        scale = self.scale_input.value()
        
        # 如果有Python端的分割器和选中的索引，使用Python端导出
        if self.splitter and self.selected_indices is not None and len(self.selected_indices) > 0:
            self.status_label.setText(f"正在导出 {len(self.selected_indices)} 个高斯点...")
            
            # 直接导出为splat格式（跳过PLY中间步骤）
            if self.splitter.export_selected_to_splat(self.selected_indices, output_splat_path):
                self.status_label.setText(f"导出成功: {output_splat_path}")
                self.load_to_cesium_requested.emit(output_splat_path, lon, lat, alt, scale)
            else:
                # 如果直接导出失败，尝试通过PLY中间步骤
                output_ply_path = f"{base}_split.ply"
                self.status_label.setText("直接导出失败，尝试PLY中间步骤...")
                if self.splitter.export_selected_points(self.selected_indices, output_ply_path):
                    if self.splitter.convert_to_splat(output_ply_path, output_splat_path):
                        self.status_label.setText(f"导出成功: {output_splat_path}")
                        self.load_to_cesium_requested.emit(output_splat_path, lon, lat, alt, scale)
                    else:
                        self.status_label.setText(f"使用PLY: {output_ply_path}")
                        self.load_to_cesium_requested.emit(output_ply_path, lon, lat, alt, scale)
                else:
                    self.status_label.setText("导出失败")
                    QMessageBox.warning(self, "警告", "导出分割模型失败")
        else:
            # 使用JavaScript端的导出（如果没有Python端分割器）
            output_path = self.viewer.export_split_model()
            if output_path:
                self.status_label.setText(f"正在导出到: {output_path}")
                self.load_to_cesium_requested.emit(output_path, lon, lat, alt, scale)
            
    def _on_model_loaded(self, success: bool, error: str):
        """模型加载完成回调"""
        if success:
            self.screenshot_btn.setEnabled(True)
            point_info = ""
            if self.splitter and self.splitter.point_count > 0:
                point_info = f" ({self.splitter.point_count} 个高斯点)"
            self.status_label.setText(f"模型加载成功{point_info}")
        else:
            self.status_label.setText(f"模型加载失败: {error}")
            self.splitter = None
            
    def _on_split_complete(self, output_path: str):
        """分割完成回调"""
        self.status_label.setText(f"模型已导出: {output_path}")


class ThreeJSSplatSplitterTab(QWidget):
    """Three.js 3DGS编辑标签页 - 完整的编辑和分割工作流界面
    
    功能类似PlayCanvas SuperSplat:
    - 拉框/套索/笔刷选择
    - 切换可见性/锁定解锁
    - 旋转/缩放/全选/反选
    - 渲染模式切换（splat/点云等）
    - 右上角3D指示轴
    """
    
    # 信号：请求加载到Cesium
    load_to_cesium_requested = pyqtSignal(str, float, float, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 编辑状态
        self.selected_indices = set()
        self.hidden_indices = set()
        self.locked_indices = set()
        
        # 当前渲染模式
        self.current_render_mode = 'splat'
        
        # 撤销/重做栈
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        
        self.init_ui()
        self.connect_edit_signals()
        
    def init_ui(self):
        """初始化UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：Three.js查看器（全功能编辑器）
        self.viewer = ThreeJSSplatViewer()
        splitter.addWidget(self.viewer)
        
        # 右侧：控制面板
        self.panel = SplatSplitterPanel()
        self.panel.setFixedWidth(320)
        self.panel.set_viewer(self.viewer)
        splitter.addWidget(self.panel)
        
        # 连接信号
        self.panel.load_to_cesium_requested.connect(self._on_load_to_cesium)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        layout.addWidget(splitter)
    
    def connect_edit_signals(self):
        """连接编辑相关信号"""
        bridge = self.viewer.bridge
        
        # 选择操作
        bridge.select_by_box.connect(self._on_select_by_box)
        bridge.select_by_lasso.connect(self._on_select_by_lasso)
        bridge.select_by_brush.connect(self._on_select_by_brush)
        bridge.select_all.connect(self._on_select_all)
        bridge.select_inverse.connect(self._on_select_inverse)
        bridge.clear_selection.connect(self._on_clear_selection)
        
        # 编辑操作
        bridge.hide_selected.connect(self._on_hide_selected)
        bridge.show_all.connect(self._on_show_all)
        bridge.lock_selected.connect(self._on_lock_selected)
        bridge.unlock_all.connect(self._on_unlock_all)
        bridge.delete_selected.connect(self._on_delete_selected)
        
        # 撤销/重做
        bridge.undo_requested.connect(self._on_undo)
        bridge.redo_requested.connect(self._on_redo)
        
        # 渲染模式
        bridge.render_mode_changed.connect(self._on_render_mode_changed)
        
        # 场景管理器相关
        bridge.request_add_model.connect(self._on_request_add_model)
        bridge.model_deselected.connect(self._on_model_deselected)
        bridge.request_point_cloud_overlay.connect(self._on_request_point_cloud_overlay)
        
    def load_model(self, file_path: str):
        """加载模型文件"""
        self.panel.current_model_path = file_path
        self.panel.path_label.setText(os.path.basename(file_path))
        
        # 清除编辑状态
        self.selected_indices.clear()
        self.hidden_indices.clear()
        self.locked_indices.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        
        # 如果是PLY文件，同时加载到Python端的分割器
        if file_path.lower().endswith('.ply'):
            self.panel.splitter = GaussianSplatSplitter()
            if not self.panel.splitter.load_ply(file_path):
                self.panel.splitter = None
        else:
            self.panel.splitter = None
        
        self.viewer.load_model(file_path)
        
    def set_coordinates(self, lon: float, lat: float, alt: float, scale: float):
        """设置地理坐标"""
        self.panel.lon_input.setValue(lon)
        self.panel.lat_input.setValue(lat)
        self.panel.alt_input.setValue(alt)
        self.panel.scale_input.setValue(scale)
    
    # ==================== 选择操作处理 ====================
    
    def _save_undo_state(self):
        """保存撤销状态"""
        state = {
            'selected': set(self.selected_indices),
            'hidden': set(self.hidden_indices),
            'locked': set(self.locked_indices)
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
    
    def _update_js_selection(self):
        """更新JavaScript端的选择状态"""
        self.viewer.update_selection(
            list(self.selected_indices),
            list(self.hidden_indices),
            list(self.locked_indices)
        )
        
        # 同时发送高亮数据（带3D坐标）
        self._send_highlight_data()
        
        # 如果在点云模式，也更新点云颜色
        if self.current_render_mode in ('points', 'centers'):
            self._send_point_cloud_data()
    
    def _on_select_by_box(self, data: dict):
        """处理框选"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        self._save_undo_state()
        
        # 从数据中提取参数
        min_x = data.get('minX', 0)
        min_y = data.get('minY', 0)
        max_x = data.get('maxX', 0)
        max_y = data.get('maxY', 0)
        screen_width = data.get('screenWidth', 1)
        screen_height = data.get('screenHeight', 1)
        add_to_selection = data.get('addToSelection', False)
        remove_from_selection = data.get('removeFromSelection', False)
        view_matrix = np.array(data.get('viewMatrix', [])).reshape(4, 4).T if data.get('viewMatrix') else None
        proj_matrix = np.array(data.get('projMatrix', [])).reshape(4, 4).T if data.get('projMatrix') else None
        
        if view_matrix is None or proj_matrix is None:
            return
        
        # 选择框内的点
        selected = self._select_points_in_screen_rect(
            min_x, min_y, max_x, max_y,
            screen_width, screen_height,
            view_matrix, proj_matrix
        )
        
        # 更新选择
        if remove_from_selection:
            self.selected_indices -= selected
        elif add_to_selection:
            self.selected_indices |= selected
        else:
            self.selected_indices = selected
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 框选完成: {len(self.selected_indices)} 个点")
    
    def _on_select_by_lasso(self, data: dict):
        """处理套索选择"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        self._save_undo_state()
        
        path = data.get('path', [])
        screen_width = data.get('screenWidth', 1)
        screen_height = data.get('screenHeight', 1)
        add_to_selection = data.get('addToSelection', False)
        remove_from_selection = data.get('removeFromSelection', False)
        view_matrix = np.array(data.get('viewMatrix', [])).reshape(4, 4).T if data.get('viewMatrix') else None
        proj_matrix = np.array(data.get('projMatrix', [])).reshape(4, 4).T if data.get('projMatrix') else None
        
        if view_matrix is None or proj_matrix is None or len(path) < 3:
            return
        
        # 选择套索内的点
        selected = self._select_points_in_lasso(
            path, screen_width, screen_height,
            view_matrix, proj_matrix
        )
        
        if remove_from_selection:
            self.selected_indices -= selected
        elif add_to_selection:
            self.selected_indices |= selected
        else:
            self.selected_indices = selected
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 套索选择完成: {len(self.selected_indices)} 个点")
    
    def _on_select_by_brush(self, data: dict):
        """处理笔刷选择"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        self._save_undo_state()
        
        path = data.get('path', [])
        radius = data.get('radius', 30)
        screen_width = data.get('screenWidth', 1)
        screen_height = data.get('screenHeight', 1)
        add_to_selection = data.get('addToSelection', False)
        remove_from_selection = data.get('removeFromSelection', False)
        view_matrix = np.array(data.get('viewMatrix', [])).reshape(4, 4).T if data.get('viewMatrix') else None
        proj_matrix = np.array(data.get('projMatrix', [])).reshape(4, 4).T if data.get('projMatrix') else None
        
        if view_matrix is None or proj_matrix is None or len(path) < 1:
            return
        
        # 选择笔刷路径上的点
        selected = self._select_points_by_brush(
            path, radius, screen_width, screen_height,
            view_matrix, proj_matrix
        )
        
        if remove_from_selection:
            self.selected_indices -= selected
        elif add_to_selection:
            self.selected_indices |= selected
        else:
            self.selected_indices = selected
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 笔刷选择完成: {len(self.selected_indices)} 个点")
    
    def _on_select_all(self):
        """全选"""
        if not self.panel.splitter:
            return
        
        self._save_undo_state()
        
        # 选择所有未隐藏的点
        all_indices = set(range(self.panel.splitter.point_count))
        self.selected_indices = all_indices - self.hidden_indices
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 全选: {len(self.selected_indices)} 个点")
    
    def _on_select_inverse(self):
        """反选"""
        if not self.panel.splitter:
            return
        
        self._save_undo_state()
        
        # 反选（排除隐藏的点）
        all_indices = set(range(self.panel.splitter.point_count))
        visible_indices = all_indices - self.hidden_indices
        self.selected_indices = visible_indices - self.selected_indices
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 反选: {len(self.selected_indices)} 个点")
    
    def _on_clear_selection(self):
        """清除选择"""
        self._save_undo_state()
        self.selected_indices.clear()
        self._update_js_selection()
    
    # ==================== 编辑操作处理 ====================
    
    def _on_hide_selected(self):
        """隐藏选中的点"""
        if not self.selected_indices:
            return
        
        self._save_undo_state()
        self.hidden_indices |= self.selected_indices
        self.selected_indices.clear()
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 隐藏: {len(self.hidden_indices)} 个点")
    
    def _on_show_all(self):
        """显示所有点"""
        self._save_undo_state()
        self.hidden_indices.clear()
        self._update_js_selection()
        print("[ThreeJSSplatSplitterTab] 显示全部")
    
    def _on_lock_selected(self):
        """锁定选中的点"""
        if not self.selected_indices:
            return
        
        self._save_undo_state()
        self.locked_indices |= self.selected_indices
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 锁定: {len(self.locked_indices)} 个点")
    
    def _on_unlock_all(self):
        """解锁所有点"""
        self._save_undo_state()
        self.locked_indices.clear()
        self._update_js_selection()
        print("[ThreeJSSplatSplitterTab] 解锁全部")
    
    def _on_delete_selected(self):
        """删除选中的点（排除锁定的点）"""
        if not self.selected_indices:
            return
        
        # 排除锁定的点
        to_delete = self.selected_indices - self.locked_indices
        if not to_delete:
            print("[ThreeJSSplatSplitterTab] 所有选中的点都被锁定，无法删除")
            return
        
        self._save_undo_state()
        
        # 标记为隐藏（实际删除在导出时处理）
        self.hidden_indices |= to_delete
        self.selected_indices.clear()
        
        self._update_js_selection()
        print(f"[ThreeJSSplatSplitterTab] 删除: {len(to_delete)} 个点")
    
    # ==================== 撤销/重做 ====================
    
    def _on_undo(self):
        """撤销"""
        if not self.undo_stack:
            return
        
        # 保存当前状态到重做栈
        self.redo_stack.append({
            'selected': set(self.selected_indices),
            'hidden': set(self.hidden_indices),
            'locked': set(self.locked_indices)
        })
        
        # 恢复上一个状态
        state = self.undo_stack.pop()
        self.selected_indices = state['selected']
        self.hidden_indices = state['hidden']
        self.locked_indices = state['locked']
        
        self._update_js_selection()
        print("[ThreeJSSplatSplitterTab] 撤销")
    
    def _on_redo(self):
        """重做"""
        if not self.redo_stack:
            return
        
        # 保存当前状态到撤销栈
        self.undo_stack.append({
            'selected': set(self.selected_indices),
            'hidden': set(self.hidden_indices),
            'locked': set(self.locked_indices)
        })
        
        # 恢复重做状态
        state = self.redo_stack.pop()
        self.selected_indices = state['selected']
        self.hidden_indices = state['hidden']
        self.locked_indices = state['locked']
        
        self._update_js_selection()
        print("[ThreeJSSplatSplitterTab] 重做")
    
    def _on_render_mode_changed(self, mode: str):
        """渲染模式改变"""
        print(f"[ThreeJSSplatSplitterTab] 渲染模式: {mode}")
        self.current_render_mode = mode
        
        # 如果切换到点云模式，发送点云数据
        if mode in ('points', 'centers') and self.panel.splitter and self.panel.splitter.xyz is not None:
            self._send_point_cloud_data()
    
    def _send_point_cloud_data(self):
        """发送点云数据到JavaScript端"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        xyz = self.panel.splitter.xyz
        colors = None
        
        # 尝试获取颜色数据
        splitter = self.panel.splitter
        
        # 优先使用直接的RGB颜色
        if hasattr(splitter, 'colors') and splitter.colors is not None:
            colors = splitter.colors.copy()
            # 归一化到0-1范围
            if colors.max() > 1.0:
                colors = colors / 255.0
        # 如果没有直接颜色，尝试从球谐系数提取
        elif hasattr(splitter, 'sh_coeffs') and splitter.sh_coeffs is not None:
            sh_coeffs = splitter.sh_coeffs
            # 从f_dc_0, f_dc_1, f_dc_2提取颜色
            # 公式: color = 0.5 + SH_C0 * f_dc
            SH_C0 = 0.28209479177387814
            if sh_coeffs.shape[1] >= 3:
                colors = np.zeros((len(sh_coeffs), 3), dtype=np.float32)
                colors[:, 0] = 0.5 + SH_C0 * sh_coeffs[:, 0]  # R
                colors[:, 1] = 0.5 + SH_C0 * sh_coeffs[:, 1]  # G
                colors[:, 2] = 0.5 + SH_C0 * sh_coeffs[:, 2]  # B
                colors = np.clip(colors, 0.0, 1.0)
                print(f"[ThreeJSSplatSplitterTab] 从SH系数提取颜色")
        
        self.viewer.send_point_cloud_data(
            xyz, colors,
            self.selected_indices,
            self.hidden_indices,
            self.locked_indices
        )
    
    def _send_highlight_data(self):
        """发送选中点的高亮数据到JavaScript端（向量化优化版）"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        if not self.selected_indices:
            # 清除高亮
            self.viewer.send_highlight_data(None, None)
            return
        
        xyz = self.panel.splitter.xyz
        
        # 获取选中点的坐标（排除隐藏的点）
        visible_selected = self.selected_indices - self.hidden_indices
        if not visible_selected:
            self.viewer.send_highlight_data(None, None)
            return
        
        # 转换为numpy数组
        selected_array = np.array(list(visible_selected), dtype=np.int32)
        
        # 过滤越界索引
        selected_array = selected_array[selected_array < len(xyz)]
        
        if len(selected_array) == 0:
            self.viewer.send_highlight_data(None, None)
            return
        
        # 采样以减少数据量
        max_points = 50000
        if len(selected_array) > max_points:
            sample_rate = len(selected_array) // max_points
            selected_array = selected_array[::sample_rate]
        
        # 获取选中点的坐标
        selected_xyz = xyz[selected_array].astype(np.float32)
        
        # 计算边界框
        min_coords = selected_xyz.min(axis=0)
        max_coords = selected_xyz.max(axis=0)
        
        bounding_box = {
            'minX': float(min_coords[0]), 'minY': float(min_coords[1]), 'minZ': float(min_coords[2]),
            'maxX': float(max_coords[0]), 'maxY': float(max_coords[1]), 'maxZ': float(max_coords[2])
        }
        
        # 展平坐标数组
        positions = selected_xyz.flatten().tolist()
        
        self.viewer.send_highlight_data(positions, bounding_box)
    
    # ==================== 选择辅助方法（性能优化版） ====================
    
    def _project_points_to_screen(self, view_matrix, proj_matrix, screen_width, screen_height):
        """
        将所有点投影到屏幕空间（缓存结果以提高性能）
        返回: (screen_x, screen_y, valid_mask)
        """
        xyz = self.panel.splitter.xyz
        n = len(xyz)
        
        # 组合MVP矩阵
        mvp = np.ascontiguousarray(proj_matrix @ view_matrix, dtype=np.float32)
        
        # 高效的齐次坐标变换（避免创建大的中间数组）
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        # 直接计算clip坐标的w分量
        clip_w = mvp[3, 0] * x + mvp[3, 1] * y + mvp[3, 2] * z + mvp[3, 3]
        
        # 只处理w > 0的点
        valid = clip_w > 0.001
        
        # 预分配结果数组
        screen_x = np.zeros(n, dtype=np.int32)
        screen_y = np.zeros(n, dtype=np.int32)
        
        if np.any(valid):
            # 只对有效点计算完整投影
            x_v, y_v, z_v = x[valid], y[valid], z[valid]
            w_v = clip_w[valid]
            inv_w = 1.0 / w_v
            
            # NDC坐标
            ndc_x = (mvp[0, 0] * x_v + mvp[0, 1] * y_v + mvp[0, 2] * z_v + mvp[0, 3]) * inv_w
            ndc_y = (mvp[1, 0] * x_v + mvp[1, 1] * y_v + mvp[1, 2] * z_v + mvp[1, 3]) * inv_w
            
            # 转换为屏幕坐标
            screen_x[valid] = ((ndc_x + 1) * 0.5 * screen_width).astype(np.int32)
            screen_y[valid] = ((1 - ndc_y) * 0.5 * screen_height).astype(np.int32)
        
        return screen_x, screen_y, valid
    
    def _select_points_in_screen_rect(self, min_x, min_y, max_x, max_y,
                                       screen_width, screen_height,
                                       view_matrix, proj_matrix) -> set:
        """选择屏幕矩形内的点（完全向量化）"""
        screen_x, screen_y, valid = self._project_points_to_screen(
            view_matrix, proj_matrix, screen_width, screen_height
        )
        
        # 检查是否在矩形内（向量化）
        in_rect = valid & (screen_x >= min_x) & (screen_x <= max_x) & \
                  (screen_y >= min_y) & (screen_y <= max_y)
        
        # 获取选中的索引
        selected_indices = np.where(in_rect)[0]
        
        # 排除隐藏的点（向量化）
        if self.hidden_indices:
            hidden_array = np.array(list(self.hidden_indices), dtype=np.int32)
            mask = ~np.isin(selected_indices, hidden_array)
            selected_indices = selected_indices[mask]
        
        return set(selected_indices.tolist())
    
    def _select_points_in_lasso(self, path, screen_width, screen_height,
                                 view_matrix, proj_matrix) -> set:
        """选择套索内的点（完全向量化）"""
        import cv2
        
        screen_x, screen_y, valid = self._project_points_to_screen(
            view_matrix, proj_matrix, screen_width, screen_height
        )
        
        # 创建套索mask
        mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
        pts = np.array([[int(p['x']), int(p['y'])] for p in path], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # 检查边界（向量化）
        in_bounds = valid & (screen_x >= 0) & (screen_x < screen_width) & \
                    (screen_y >= 0) & (screen_y < screen_height)
        
        # 获取边界内的索引
        bound_indices = np.where(in_bounds)[0]
        
        if len(bound_indices) == 0:
            return set()
        
        # 批量检查mask值（向量化）
        sx = screen_x[bound_indices]
        sy = screen_y[bound_indices]
        mask_values = mask[sy, sx]
        
        # 选中mask内的点
        in_mask = mask_values > 0
        selected_indices = bound_indices[in_mask]
        
        # 排除隐藏的点
        if self.hidden_indices:
            hidden_array = np.array(list(self.hidden_indices), dtype=np.int32)
            mask_hidden = ~np.isin(selected_indices, hidden_array)
            selected_indices = selected_indices[mask_hidden]
        
        return set(selected_indices.tolist())
    
    def _select_points_by_brush(self, path, radius, screen_width, screen_height,
                                 view_matrix, proj_matrix) -> set:
        """选择笔刷路径上的点（完全向量化）"""
        import cv2
        
        screen_x, screen_y, valid = self._project_points_to_screen(
            view_matrix, proj_matrix, screen_width, screen_height
        )
        
        # 创建笔刷mask（优化：使用polylines代替多个circle）
        mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
        if len(path) > 1:
            pts = np.array([[int(p['x']), int(p['y'])] for p in path], dtype=np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness=int(radius * 2))
        else:
            for p in path:
                cv2.circle(mask, (int(p['x']), int(p['y'])), int(radius), 255, -1)
        
        # 检查边界（向量化）
        in_bounds = valid & (screen_x >= 0) & (screen_x < screen_width) & \
                    (screen_y >= 0) & (screen_y < screen_height)
        
        bound_indices = np.where(in_bounds)[0]
        
        if len(bound_indices) == 0:
            return set()
        
        # 批量检查mask值
        sx = screen_x[bound_indices]
        sy = screen_y[bound_indices]
        mask_values = mask[sy, sx]
        
        in_mask = mask_values > 0
        selected_indices = bound_indices[in_mask]
        
        # 排除隐藏的点
        if self.hidden_indices:
            hidden_array = np.array(list(self.hidden_indices), dtype=np.int32)
            mask_hidden = ~np.isin(selected_indices, hidden_array)
            selected_indices = selected_indices[mask_hidden]
        
        return set(selected_indices.tolist())
        
    def _on_load_to_cesium(self, path: str, lon: float, lat: float, alt: float, scale: float):
        """转发加载到Cesium的请求"""
        self.load_to_cesium_requested.emit(path, lon, lat, alt, scale)
    
    def _on_request_add_model(self):
        """处理添加模型请求"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择3DGS模型文件", "", 
            "Gaussian Splat Files (*.ply *.splat *.ksplat);;All Files (*)"
        )
        if file_path:
            self.load_model(file_path)
    
    def _on_model_deselected(self):
        """处理取消选中"""
        self.selected_indices.clear()
        self._update_js_selection()
        print("[ThreeJSSplatSplitterTab] 取消选中")
    
    def _on_request_point_cloud_overlay(self):
        """处理点云叠加请求"""
        if not self.panel.splitter or self.panel.splitter.xyz is None:
            return
        
        xyz = self.panel.splitter.xyz
        colors = None
        
        # 尝试获取颜色数据
        splitter = self.panel.splitter
        if hasattr(splitter, 'colors') and splitter.colors is not None:
            colors = splitter.colors.copy()
            if colors.max() > 1.0:
                colors = colors / 255.0
        elif hasattr(splitter, 'sh_coeffs') and splitter.sh_coeffs is not None:
            sh_coeffs = splitter.sh_coeffs
            SH_C0 = 0.28209479177387814
            if sh_coeffs.shape[1] >= 3:
                colors = np.zeros((len(sh_coeffs), 3), dtype=np.float32)
                colors[:, 0] = 0.5 + SH_C0 * sh_coeffs[:, 0]
                colors[:, 1] = 0.5 + SH_C0 * sh_coeffs[:, 1]
                colors[:, 2] = 0.5 + SH_C0 * sh_coeffs[:, 2]
                colors = np.clip(colors, 0.0, 1.0)
        
        self.viewer.send_point_cloud_overlay(xyz, colors)
        print(f"[ThreeJSSplatSplitterTab] 发送点云叠加数据: {len(xyz)} 个点")


class GaussianSplatSplitter:
    """高斯点分割器 - 在Python端处理PLY文件的分割和导出"""
    
    def __init__(self):
        self.ply_data = None
        self.xyz = None
        self.colors = None
        self.opacities = None
        self.scales = None
        self.rotations = None
        self.sh_coeffs = None
        self.point_count = 0
        
    def load_ply(self, file_path: str) -> bool:
        """
        加载PLY文件（支持标准和压缩格式）
        
        Args:
            file_path: PLY文件路径
            
        Returns:
            是否加载成功
        """
        try:
            from plyfile import PlyData
            
            self.ply_data = PlyData.read(file_path)
            vertex = self.ply_data['vertex']
            
            # 获取所有字段名
            field_names = vertex.data.dtype.names
            print(f"[GaussianSplatSplitter] PLY字段: {field_names}")
            
            # 检查是否是压缩格式（packed_position等）
            if 'packed_position' in field_names:
                print("[GaussianSplatSplitter] 检测到压缩格式PLY，尝试解压...")
                return self._load_compressed_ply(vertex, field_names)
            
            # 提取坐标 - 支持多种字段名格式
            x_field = 'x' if 'x' in field_names else 'position_x' if 'position_x' in field_names else None
            y_field = 'y' if 'y' in field_names else 'position_y' if 'position_y' in field_names else None
            z_field = 'z' if 'z' in field_names else 'position_z' if 'position_z' in field_names else None
            
            # 如果找不到标准字段，尝试查找包含position的字段
            if x_field is None:
                for name in field_names:
                    if 'x' in name.lower() and 'packed' not in name.lower():
                        x_field = name
                        break
            if y_field is None:
                for name in field_names:
                    if 'y' in name.lower() and 'packed' not in name.lower():
                        y_field = name
                        break
            if z_field is None:
                for name in field_names:
                    if 'z' in name.lower() and 'packed' not in name.lower():
                        z_field = name
                        break
            
            if x_field is None or y_field is None or z_field is None:
                # 尝试使用前三个数值字段作为坐标
                numeric_fields = [name for name in field_names 
                                  if vertex[name].dtype in [np.float32, np.float64, np.int32, np.int64]
                                  and 'packed' not in name.lower()]
                if len(numeric_fields) >= 3:
                    x_field, y_field, z_field = numeric_fields[0], numeric_fields[1], numeric_fields[2]
                    print(f"[GaussianSplatSplitter] 使用字段作为坐标: {x_field}, {y_field}, {z_field}")
                else:
                    raise ValueError(f"无法找到坐标字段，可用字段: {field_names}")
            
            self.xyz = np.column_stack([
                vertex[x_field],
                vertex[y_field],
                vertex[z_field]
            ])
            
            self.point_count = len(self.xyz)
            
            # 提取颜色（如果存在）
            if 'red' in vertex.data.dtype.names:
                self.colors = np.column_stack([
                    vertex['red'],
                    vertex['green'],
                    vertex['blue']
                ]).astype(np.uint8)  # 确保是uint8类型
            elif 'f_dc_0' in vertex.data.dtype.names:
                # 从球谐系数DC项计算RGB颜色
                # 公式: RGB = 0.5 + SH_C0 * f_dc
                SH_C0 = 0.28209479177387814
                colors_r = 0.5 + SH_C0 * vertex['f_dc_0']
                colors_g = 0.5 + SH_C0 * vertex['f_dc_1']
                colors_b = 0.5 + SH_C0 * vertex['f_dc_2']
                colors = np.stack([colors_r, colors_g, colors_b], axis=1)
                colors = np.clip(colors, 0.0, 1.0) * 255.0
                self.colors = colors.astype(np.uint8)
                print(f"[GaussianSplatSplitter] 从球谐系数提取颜色: {self.colors.shape}")
            
            # 提取不透明度（如果存在）
            if 'opacity' in vertex.data.dtype.names:
                self.opacities = vertex['opacity']
            
            # 提取缩放（如果存在）
            if 'scale_0' in vertex.data.dtype.names:
                self.scales = np.column_stack([
                    vertex['scale_0'],
                    vertex['scale_1'],
                    vertex['scale_2']
                ])
            
            # 提取旋转（如果存在）
            if 'rot_0' in vertex.data.dtype.names:
                self.rotations = np.column_stack([
                    vertex['rot_0'],
                    vertex['rot_1'],
                    vertex['rot_2'],
                    vertex['rot_3']
                ])
            
            # 提取球谐系数（如果存在）
            sh_names = [name for name in vertex.data.dtype.names if name.startswith('f_dc_') or name.startswith('f_rest_')]
            if sh_names:
                self.sh_coeffs = np.column_stack([vertex[name] for name in sh_names])
            
            print(f"[GaussianSplatSplitter] 加载PLY成功: {self.point_count} 个高斯点")
            return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 加载PLY失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_compressed_ply(self, vertex, field_names) -> bool:
        """
        加载压缩格式的PLY文件（packed_position等字段）
        
        压缩格式由 splat-transform 工具生成，使用 chunk 边界框进行量化压缩。
        packed_position 是一个 uint32，编码为: x(10位) + y(10位) + z(10位) + unused(2位)
        每个chunk包含256个顶点，按顺序排列。
        
        Args:
            vertex: PLY顶点数据
            field_names: 字段名列表
            
        Returns:
            是否加载成功
        """
        try:
            point_count = len(vertex)
            
            # 检查packed_position的格式
            if 'packed_position' not in field_names:
                raise ValueError("找不到packed_position字段")
            
            packed_pos = vertex['packed_position']
            print(f"[GaussianSplatSplitter] packed_position dtype: {packed_pos.dtype}, shape: {packed_pos.shape}")
            
            if packed_pos.dtype != np.uint32:
                # 非uint32格式，尝试其他方式
                if len(packed_pos.shape) > 1 and packed_pos.shape[1] >= 3:
                    self.xyz = packed_pos[:, :3].astype(np.float32)
                elif packed_pos.dtype in [np.float32, np.float64]:
                    if len(packed_pos) == point_count * 3:
                        self.xyz = packed_pos.reshape(point_count, 3).astype(np.float32)
                    else:
                        self.xyz = np.zeros((point_count, 3), dtype=np.float32)
                        self._is_compressed = True
                else:
                    self.xyz = np.zeros((point_count, 3), dtype=np.float32)
                    self._is_compressed = True
                self.point_count = len(self.xyz)
                return True
            
            # 尝试从PLY数据中获取chunk信息
            chunks = None
            if hasattr(self, 'ply_data') and self.ply_data is not None:
                for element in self.ply_data.elements:
                    if element.name == 'chunk':
                        chunks = element.data
                        print(f"[GaussianSplatSplitter] 找到 {len(chunks)} 个chunk")
                        break
            
            if chunks is None or len(chunks) == 0:
                print("[GaussianSplatSplitter] 警告：未找到chunk数据，无法解码压缩坐标")
                self._is_compressed = True
                self.xyz = np.zeros((point_count, 3), dtype=np.float32)
                self.point_count = point_count
                return True
            
            # 解码压缩坐标
            print("[GaussianSplatSplitter] 开始解码压缩坐标...")
            
            num_chunks = len(chunks)
            POINTS_PER_CHUNK = 256  # splat-transform 默认每个chunk 256个点
            
            # 10-10-10-2 位编码
            BITS_PER_AXIS = 10
            MAX_VAL = (1 << BITS_PER_AXIS) - 1  # 1023
            
            # 提取各轴的量化值（向量化操作）
            packed = packed_pos.astype(np.uint32)
            x_quant = (packed >> 0) & MAX_VAL
            y_quant = (packed >> 10) & MAX_VAL
            z_quant = (packed >> 20) & MAX_VAL
            
            # 根据顶点索引确定chunk（每256个点一个chunk）
            chunk_indices = np.arange(point_count) // POINTS_PER_CHUNK
            chunk_indices = np.clip(chunk_indices, 0, num_chunks - 1)
            
            # 获取每个chunk的边界框（向量化）
            min_x = np.array([chunks[i]['min_x'] for i in range(num_chunks)], dtype=np.float32)
            min_y = np.array([chunks[i]['min_y'] for i in range(num_chunks)], dtype=np.float32)
            min_z = np.array([chunks[i]['min_z'] for i in range(num_chunks)], dtype=np.float32)
            max_x = np.array([chunks[i]['max_x'] for i in range(num_chunks)], dtype=np.float32)
            max_y = np.array([chunks[i]['max_y'] for i in range(num_chunks)], dtype=np.float32)
            max_z = np.array([chunks[i]['max_z'] for i in range(num_chunks)], dtype=np.float32)
            
            # 反量化坐标（向量化操作，高效）
            xyz = np.zeros((point_count, 3), dtype=np.float32)
            xyz[:, 0] = min_x[chunk_indices] + (x_quant / MAX_VAL) * (max_x[chunk_indices] - min_x[chunk_indices])
            xyz[:, 1] = min_y[chunk_indices] + (y_quant / MAX_VAL) * (max_y[chunk_indices] - min_y[chunk_indices])
            xyz[:, 2] = min_z[chunk_indices] + (z_quant / MAX_VAL) * (max_z[chunk_indices] - min_z[chunk_indices])
            
            self.xyz = xyz
            self.point_count = point_count
            self._is_compressed = False  # 已成功解码
            
            # 验证解码结果
            x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
            y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
            z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
            print(f"[GaussianSplatSplitter] 解码后坐标范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
            
            # 尝试解析颜色
            if 'packed_color' in field_names:
                packed_color = vertex['packed_color'].astype(np.uint32)
                # 颜色编码为 R(8bit) + G(8bit) + B(8bit) + A(8bit)
                r = (packed_color >> 0) & 0xFF
                g = (packed_color >> 8) & 0xFF
                b = (packed_color >> 16) & 0xFF
                self.colors = np.column_stack([r, g, b]).astype(np.uint8)
            elif 'f_dc_0' in field_names:
                # 从球谐系数DC项计算RGB颜色
                SH_C0 = 0.28209479177387814
                colors_r = 0.5 + SH_C0 * vertex['f_dc_0']
                colors_g = 0.5 + SH_C0 * vertex['f_dc_1']
                colors_b = 0.5 + SH_C0 * vertex['f_dc_2']
                colors = np.stack([colors_r, colors_g, colors_b], axis=1)
                colors = np.clip(colors, 0.0, 1.0) * 255.0
                self.colors = colors.astype(np.uint8)
                print(f"[GaussianSplatSplitter] 压缩格式：从球谐系数提取颜色")
            
            print(f"[GaussianSplatSplitter] 压缩PLY解码成功: {self.point_count} 个高斯点")
            return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解压PLY失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败时创建占位数据
            self.xyz = np.zeros((point_count, 3), dtype=np.float32)
            self.point_count = point_count
            self._is_compressed = True
            return True  # 仍然返回True，允许可视化
    
    def select_points_by_screen_mask(self, mask: np.ndarray, 
                                      view_matrix: np.ndarray, 
                                      proj_matrix: np.ndarray,
                                      image_width: int, 
                                      image_height: int,
                                      keep_inside: bool = True) -> np.ndarray:
        """
        根据屏幕空间mask选择高斯点
        
        Args:
            mask: 分割mask (height, width)
            view_matrix: 视图矩阵 (4x4)
            proj_matrix: 投影矩阵 (4x4)
            image_width: 图像宽度
            image_height: 图像高度
            keep_inside: True保留mask内的点，False保留mask外的点
            
        Returns:
            选中点的索引数组
        """
        if self.xyz is None:
            return np.array([], dtype=np.int32)
        
        selected_indices = []
        
        # 组合视图和投影矩阵
        mvp = proj_matrix @ view_matrix
        
        for i in range(self.point_count):
            # 获取点的世界坐标
            world_pos = np.array([self.xyz[i, 0], self.xyz[i, 1], self.xyz[i, 2], 1.0])
            
            # 投影到裁剪空间
            clip_pos = mvp @ world_pos
            
            # 透视除法
            if abs(clip_pos[3]) < 1e-6:
                continue
            ndc = clip_pos[:3] / clip_pos[3]
            
            # 检查是否在视锥体内
            if ndc[0] < -1 or ndc[0] > 1 or ndc[1] < -1 or ndc[1] > 1 or ndc[2] < -1 or ndc[2] > 1:
                continue
            
            # 转换为屏幕坐标
            screen_x = int((ndc[0] + 1) / 2 * image_width)
            screen_y = int((1 - ndc[1]) / 2 * image_height)
            
            # 检查是否在图像范围内
            if screen_x < 0 or screen_x >= image_width or screen_y < 0 or screen_y >= image_height:
                continue
            
            # 检查mask值
            mask_value = mask[screen_y, screen_x]
            
            if keep_inside:
                if mask_value > 0:
                    selected_indices.append(i)
            else:
                if mask_value == 0:
                    selected_indices.append(i)
        
        return np.array(selected_indices, dtype=np.int32)
    
    def export_selected_points(self, indices: np.ndarray, output_path: str) -> bool:
        """
        导出选中的高斯点到新的PLY文件（标准格式）
        
        Args:
            indices: 选中点的索引数组
            output_path: 输出文件路径
            
        Returns:
            是否导出成功
        """
        if self.ply_data is None or len(indices) == 0:
            return False
        
        try:
            from plyfile import PlyData, PlyElement
            
            vertex = self.ply_data['vertex']
            field_names = vertex.data.dtype.names
            
            # 检查是否是压缩格式
            if 'packed_position' in field_names:
                # 压缩格式：需要转换为标准格式
                return self._export_compressed_to_standard(indices, output_path)
            else:
                # 标准格式：直接导出选中的点
                new_vertex_data = vertex.data[indices]
                new_vertex = PlyElement.describe(new_vertex_data, 'vertex')
                new_ply = PlyData([new_vertex])
                new_ply.write(output_path)
                print(f"[GaussianSplatSplitter] 导出成功: {len(indices)} 个点 -> {output_path}")
                return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _export_compressed_to_standard(self, indices: np.ndarray, output_path: str) -> bool:
        """
        将压缩格式PLY的选中点导出为标准格式PLY
        
        Args:
            indices: 选中点的索引数组
            output_path: 输出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            from plyfile import PlyData, PlyElement
            
            # 获取选中点的解码坐标
            selected_xyz = self.xyz[indices]
            
            # 获取颜色（如果有）
            selected_colors = None
            if self.colors is not None:
                selected_colors = self.colors[indices]
            
            num_points = len(indices)
            
            # 创建标准格式的PLY数据
            # 基本属性：x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
            dtype = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
            ]
            
            vertex_data = np.zeros(num_points, dtype=dtype)
            
            # 填充坐标
            vertex_data['x'] = selected_xyz[:, 0]
            vertex_data['y'] = selected_xyz[:, 1]
            vertex_data['z'] = selected_xyz[:, 2]
            
            # 法线（默认为0）
            vertex_data['nx'] = 0
            vertex_data['ny'] = 0
            vertex_data['nz'] = 0
            
            # 颜色（从packed_color解码或使用默认值）
            if selected_colors is not None:
                # 将RGB转换为球谐系数DC项
                # SH_C0 = 0.28209479177387814
                # color = 0.5 + SH_C0 * f_dc
                # f_dc = (color - 0.5) / SH_C0
                SH_C0 = 0.28209479177387814
                vertex_data['f_dc_0'] = (selected_colors[:, 0] / 255.0 - 0.5) / SH_C0
                vertex_data['f_dc_1'] = (selected_colors[:, 1] / 255.0 - 0.5) / SH_C0
                vertex_data['f_dc_2'] = (selected_colors[:, 2] / 255.0 - 0.5) / SH_C0
            else:
                vertex_data['f_dc_0'] = 0
                vertex_data['f_dc_1'] = 0
                vertex_data['f_dc_2'] = 0
            
            # 不透明度（默认值，sigmoid逆变换后约为0）
            vertex_data['opacity'] = 0  # sigmoid(0) = 0.5
            
            # 缩放（默认值）
            vertex_data['scale_0'] = -5.0  # exp(-5) ≈ 0.007
            vertex_data['scale_1'] = -5.0
            vertex_data['scale_2'] = -5.0
            
            # 旋转（单位四元数）
            vertex_data['rot_0'] = 1.0  # w
            vertex_data['rot_1'] = 0.0  # x
            vertex_data['rot_2'] = 0.0  # y
            vertex_data['rot_3'] = 0.0  # z
            
            # 尝试从原始压缩数据中解码更多属性
            vertex = self.ply_data['vertex']
            if 'packed_scale' in vertex.data.dtype.names and 'packed_rotation' in vertex.data.dtype.names:
                self._decode_scale_rotation(vertex, indices, vertex_data)
            
            if 'packed_color' in vertex.data.dtype.names:
                self._decode_opacity_from_color(vertex, indices, vertex_data)
            
            # 创建PLY文件
            new_vertex = PlyElement.describe(vertex_data, 'vertex')
            new_ply = PlyData([new_vertex])
            new_ply.write(output_path)
            
            print(f"[GaussianSplatSplitter] 压缩格式转标准格式导出成功: {num_points} 个点 -> {output_path}")
            return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 压缩格式导出失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _decode_scale_rotation(self, vertex, indices, vertex_data):
        """从packed_scale和packed_rotation解码缩放和旋转"""
        try:
            # 获取chunk数据
            chunks = None
            for element in self.ply_data.elements:
                if element.name == 'chunk':
                    chunks = element.data
                    break
            
            if chunks is None:
                return
            
            num_chunks = len(chunks)
            POINTS_PER_CHUNK = 256
            BITS = 10
            MAX_VAL = (1 << BITS) - 1
            
            packed_scale = vertex['packed_scale'][indices].astype(np.uint32)
            packed_rot = vertex['packed_rotation'][indices].astype(np.uint32)
            
            # 确定每个点的chunk索引
            chunk_indices = indices // POINTS_PER_CHUNK
            chunk_indices = np.clip(chunk_indices, 0, num_chunks - 1)
            
            # 解码缩放
            sx_quant = (packed_scale >> 0) & MAX_VAL
            sy_quant = (packed_scale >> 10) & MAX_VAL
            sz_quant = (packed_scale >> 20) & MAX_VAL
            
            # 获取缩放边界框
            min_sx = np.array([chunks[i]['min_scale_x'] for i in range(num_chunks)])
            min_sy = np.array([chunks[i]['min_scale_y'] for i in range(num_chunks)])
            min_sz = np.array([chunks[i]['min_scale_z'] for i in range(num_chunks)])
            max_sx = np.array([chunks[i]['max_scale_x'] for i in range(num_chunks)])
            max_sy = np.array([chunks[i]['max_scale_y'] for i in range(num_chunks)])
            max_sz = np.array([chunks[i]['max_scale_z'] for i in range(num_chunks)])
            
            vertex_data['scale_0'] = min_sx[chunk_indices] + (sx_quant / MAX_VAL) * (max_sx[chunk_indices] - min_sx[chunk_indices])
            vertex_data['scale_1'] = min_sy[chunk_indices] + (sy_quant / MAX_VAL) * (max_sy[chunk_indices] - min_sy[chunk_indices])
            vertex_data['scale_2'] = min_sz[chunk_indices] + (sz_quant / MAX_VAL) * (max_sz[chunk_indices] - min_sz[chunk_indices])
            
            # 解码旋转（8-8-8-8位编码）
            r0 = ((packed_rot >> 0) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r1 = ((packed_rot >> 8) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r2 = ((packed_rot >> 16) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r3 = ((packed_rot >> 24) & 0xFF).astype(np.float32) / 127.5 - 1.0
            
            # 归一化四元数
            norm = np.sqrt(r0**2 + r1**2 + r2**2 + r3**2) + 1e-8
            vertex_data['rot_0'] = r0 / norm
            vertex_data['rot_1'] = r1 / norm
            vertex_data['rot_2'] = r2 / norm
            vertex_data['rot_3'] = r3 / norm
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解码缩放/旋转失败: {e}")
    
    def _decode_opacity_from_color(self, vertex, indices, vertex_data):
        """从packed_color解码不透明度"""
        try:
            packed_color = vertex['packed_color'][indices].astype(np.uint32)
            # Alpha通道在最高8位
            alpha = ((packed_color >> 24) & 0xFF).astype(np.float32) / 255.0
            # 转换为logit（sigmoid逆变换）
            alpha = np.clip(alpha, 0.001, 0.999)
            vertex_data['opacity'] = np.log(alpha / (1 - alpha))
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解码不透明度失败: {e}")
    
    def convert_to_splat(self, ply_path: str, splat_path: str) -> bool:
        """
        将PLY文件转换为.splat格式
        
        Args:
            ply_path: 输入PLY文件路径
            splat_path: 输出.splat文件路径
            
        Returns:
            是否转换成功
        """
        try:
            # 尝试使用现有的转换器
            from ply_to_splat_converter import convert_ply_to_splat
            return convert_ply_to_splat(ply_path, splat_path)
        except Exception as e:
            print(f"[GaussianSplatSplitter] ply_to_splat_converter失败: {e}")
            print("[GaussianSplatSplitter] 尝试简单转换...")
            # 简单的转换实现
            return self._simple_ply_to_splat(ply_path, splat_path)
    
    def export_selected_to_splat(self, indices: np.ndarray, output_path: str) -> bool:
        """
        直接将选中的点导出为.splat格式（跳过PLY中间步骤）
        
        Args:
            indices: 选中点的索引数组
            output_path: 输出.splat文件路径
            
        Returns:
            是否导出成功
        """
        if self.xyz is None or len(indices) == 0:
            return False
        
        try:
            import struct
            
            # 获取选中点的数据
            selected_xyz = self.xyz[indices]
            num_points = len(indices)
            
            # 获取颜色
            if self.colors is not None:
                selected_colors = self.colors[indices]
            else:
                selected_colors = np.full((num_points, 3), 128, dtype=np.uint8)
            
            # 尝试解码缩放和旋转
            scales = np.full((num_points, 3), 0.01, dtype=np.float32)
            rotations = np.zeros((num_points, 4), dtype=np.float32)
            rotations[:, 0] = 1.0  # 单位四元数
            opacities = np.full(num_points, 200, dtype=np.uint8)  # 默认不透明度
            
            # 尝试从压缩数据解码
            if self.ply_data is not None:
                vertex = self.ply_data['vertex']
                field_names = vertex.data.dtype.names
                
                if 'packed_scale' in field_names:
                    self._decode_scale_for_splat(vertex, indices, scales)
                
                if 'packed_rotation' in field_names:
                    self._decode_rotation_for_splat(vertex, indices, rotations)
                
                if 'packed_color' in field_names:
                    self._decode_opacity_for_splat(vertex, indices, opacities)
            
            # 写入splat文件
            # 格式：pos(12) + scale(12) + color(4) + rot(4) = 32 bytes per splat
            with open(output_path, 'wb') as f:
                for i in range(num_points):
                    # 位置 (3 floats = 12 bytes)
                    f.write(struct.pack('fff', 
                        float(selected_xyz[i, 0]),
                        float(selected_xyz[i, 1]),
                        float(selected_xyz[i, 2])
                    ))
                    
                    # 缩放 (3 floats = 12 bytes)
                    f.write(struct.pack('fff',
                        float(scales[i, 0]),
                        float(scales[i, 1]),
                        float(scales[i, 2])
                    ))
                    
                    # 颜色 RGBA (4 bytes)
                    f.write(struct.pack('BBBB',
                        int(selected_colors[i, 0]),
                        int(selected_colors[i, 1]),
                        int(selected_colors[i, 2]),
                        int(opacities[i])
                    ))
                    
                    # 旋转 (4 bytes) - 转换为uint8
                    rot = rotations[i]
                    rot_uint8 = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
                    # 顺序: Z, W, X, Y (GaussianSplats3D格式)
                    f.write(struct.pack('BBBB',
                        rot_uint8[3],  # Z
                        rot_uint8[0],  # W
                        rot_uint8[1],  # X
                        rot_uint8[2]   # Y
                    ))
            
            print(f"[GaussianSplatSplitter] 直接导出splat成功: {num_points} 个点 -> {output_path}")
            return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 直接导出splat失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _decode_scale_for_splat(self, vertex, indices, scales):
        """为splat导出解码缩放"""
        try:
            chunks = None
            for element in self.ply_data.elements:
                if element.name == 'chunk':
                    chunks = element.data
                    break
            
            if chunks is None:
                return
            
            num_chunks = len(chunks)
            POINTS_PER_CHUNK = 256
            BITS = 10
            MAX_VAL = (1 << BITS) - 1
            
            packed_scale = vertex['packed_scale'][indices].astype(np.uint32)
            chunk_indices = np.clip(indices // POINTS_PER_CHUNK, 0, num_chunks - 1)
            
            sx_quant = (packed_scale >> 0) & MAX_VAL
            sy_quant = (packed_scale >> 10) & MAX_VAL
            sz_quant = (packed_scale >> 20) & MAX_VAL
            
            min_sx = np.array([chunks[i]['min_scale_x'] for i in range(num_chunks)])
            min_sy = np.array([chunks[i]['min_scale_y'] for i in range(num_chunks)])
            min_sz = np.array([chunks[i]['min_scale_z'] for i in range(num_chunks)])
            max_sx = np.array([chunks[i]['max_scale_x'] for i in range(num_chunks)])
            max_sy = np.array([chunks[i]['max_scale_y'] for i in range(num_chunks)])
            max_sz = np.array([chunks[i]['max_scale_z'] for i in range(num_chunks)])
            
            scales[:, 0] = min_sx[chunk_indices] + (sx_quant / MAX_VAL) * (max_sx[chunk_indices] - min_sx[chunk_indices])
            scales[:, 1] = min_sy[chunk_indices] + (sy_quant / MAX_VAL) * (max_sy[chunk_indices] - min_sy[chunk_indices])
            scales[:, 2] = min_sz[chunk_indices] + (sz_quant / MAX_VAL) * (max_sz[chunk_indices] - min_sz[chunk_indices])
            
            # 缩放值通常是log空间的，需要exp
            scales[:] = np.exp(scales)
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解码缩放失败: {e}")
    
    def _decode_rotation_for_splat(self, vertex, indices, rotations):
        """为splat导出解码旋转"""
        try:
            packed_rot = vertex['packed_rotation'][indices].astype(np.uint32)
            
            r0 = ((packed_rot >> 0) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r1 = ((packed_rot >> 8) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r2 = ((packed_rot >> 16) & 0xFF).astype(np.float32) / 127.5 - 1.0
            r3 = ((packed_rot >> 24) & 0xFF).astype(np.float32) / 127.5 - 1.0
            
            norm = np.sqrt(r0**2 + r1**2 + r2**2 + r3**2) + 1e-8
            rotations[:, 0] = r0 / norm
            rotations[:, 1] = r1 / norm
            rotations[:, 2] = r2 / norm
            rotations[:, 3] = r3 / norm
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解码旋转失败: {e}")
    
    def _decode_opacity_for_splat(self, vertex, indices, opacities):
        """为splat导出解码不透明度"""
        try:
            packed_color = vertex['packed_color'][indices].astype(np.uint32)
            opacities[:] = ((packed_color >> 24) & 0xFF).astype(np.uint8)
        except Exception as e:
            print(f"[GaussianSplatSplitter] 解码不透明度失败: {e}")
    
    def _simple_ply_to_splat(self, ply_path: str, splat_path: str) -> bool:
        """简单的PLY到splat转换"""
        try:
            from plyfile import PlyData
            import struct
            
            ply_data = PlyData.read(ply_path)
            vertex = ply_data['vertex']
            
            with open(splat_path, 'wb') as f:
                for i in range(len(vertex)):
                    # 位置 (3 floats)
                    x = float(vertex['x'][i])
                    y = float(vertex['y'][i])
                    z = float(vertex['z'][i])
                    
                    # 缩放 (3 floats)
                    if 'scale_0' in vertex.data.dtype.names:
                        sx = float(vertex['scale_0'][i])
                        sy = float(vertex['scale_1'][i])
                        sz = float(vertex['scale_2'][i])
                    else:
                        sx = sy = sz = 0.01
                    
                    # 颜色 (4 bytes: RGBA)
                    if 'red' in vertex.data.dtype.names:
                        r = int(vertex['red'][i])
                        g = int(vertex['green'][i])
                        b = int(vertex['blue'][i])
                    else:
                        r = g = b = 128
                    
                    if 'opacity' in vertex.data.dtype.names:
                        a = int(min(255, max(0, (1 / (1 + np.exp(-float(vertex['opacity'][i])))) * 255)))
                    else:
                        a = 255
                    
                    # 旋转 (4 bytes: quaternion as uint8)
                    if 'rot_0' in vertex.data.dtype.names:
                        rot = np.array([
                            float(vertex['rot_0'][i]),
                            float(vertex['rot_1'][i]),
                            float(vertex['rot_2'][i]),
                            float(vertex['rot_3'][i])
                        ])
                        rot = rot / (np.linalg.norm(rot) + 1e-8)
                        rot_bytes = ((rot * 0.5 + 0.5) * 255).astype(np.uint8)
                    else:
                        rot_bytes = np.array([128, 128, 128, 255], dtype=np.uint8)
                    
                    # 写入数据
                    f.write(struct.pack('fff', x, y, z))
                    f.write(struct.pack('fff', sx, sy, sz))
                    f.write(struct.pack('BBBB', r, g, b, a))
                    f.write(struct.pack('BBBB', *rot_bytes))
            
            print(f"[GaussianSplatSplitter] 转换成功: {ply_path} -> {splat_path}")
            return True
            
        except Exception as e:
            print(f"[GaussianSplatSplitter] 转换失败: {e}")
            import traceback
            traceback.print_exc()
            return False
