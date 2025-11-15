#!/usr/bin/env python3
"""
Cesium Widget for PyQt5
使用PyQtWebEngine集成Cesium到PyQt5应用中
"""

import os
import json
from typing import Optional, Dict, Any
from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
from PyQt5.QtWebChannel import QWebChannel


class PyBridge(QObject):
    """Python和JavaScript之间的桥接对象"""
    
    # 定义信号
    message_received = pyqtSignal(str)  # 从JS接收消息
    viewer_ready = pyqtSignal()  # Cesium viewer准备就绪
    load_complete = pyqtSignal(bool, str)  # 加载完成 (success, message)
    object_clicked = pyqtSignal(dict)  # 对象被点击
    edit_mode_changed = pyqtSignal(bool)  # 编辑模式改变
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    @pyqtSlot(str)
    def receiveMessage(self, message: str):
        """
        从JavaScript接收消息
        
        Args:
            message: JSON格式的消息字符串
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            print(f"[PyBridge] 收到消息: {msg_type}")
            
            if msg_type == 'viewerReady':
                self.viewer_ready.emit()
            elif msg_type == 'loadComplete':
                success = data.get('success', False)
                error = data.get('error', '')
                self.load_complete.emit(success, error)
            elif msg_type == 'objectClicked':
                self.object_clicked.emit(data)
            elif msg_type == 'editModeChanged':
                enabled = data.get('enabled', False)
                self.edit_mode_changed.emit(enabled)
            else:
                self.message_received.emit(message)
                
        except json.JSONDecodeError as e:
            print(f"[PyBridge] JSON解析错误: {e}")
        except Exception as e:
            print(f"[PyBridge] 处理消息时出错: {e}")


class CesiumWebPage(QWebEnginePage):
    """自定义Web页面类，用于处理控制台消息"""
    
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        """
        处理JavaScript控制台消息
        
        Args:
            level: 消息级别
            message: 消息内容
            lineNumber: 行号
            sourceID: 源文件
        """
        level_name = ['Info', 'Warning', 'Error'][level]
        print(f"[Cesium JS {level_name}] {message} (line {lineNumber})")


class CesiumWidget(QWidget):
    """Cesium地图查看器Widget"""
    
    # 定义信号
    viewer_ready = pyqtSignal()
    load_complete = pyqtSignal(bool, str)
    object_clicked = pyqtSignal(dict)
    
    def __init__(self, html_path: Optional[str] = None, parent=None):
        """
        初始化Cesium Widget
        
        Args:
            html_path: Cesium HTML文件路径，如果为None则使用默认路径
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 设置HTML路径
        if html_path is None:
            html_path = os.path.join(os.path.dirname(__file__), 'cesium_viewer.html')
        self.html_path = html_path
        
        # 初始化UI
        self.init_ui()
        
        # 加载HTML
        self.load_cesium()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建Web视图
        self.web_view = QWebEngineView()
        
        # 配置WebEngine设置，允许本地文件访问远程资源
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent, True)
        # 启用JavaScript
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        # 启用本地存储
        settings.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        
        # 使用自定义页面类
        page = CesiumWebPage(self.web_view)
        self.web_view.setPage(page)
        
        # 在页面加载后再次设置，确保设置生效
        self.web_view.loadFinished.connect(self.on_load_finished)
        
        # 设置Web Channel用于Python-JS通信
        self.channel = QWebChannel()
        self.bridge = PyBridge(self)
        self.channel.registerObject('pyBridge', self.bridge)
        page.setWebChannel(self.channel)
        
        # 连接桥接信号
        self.bridge.viewer_ready.connect(self.on_viewer_ready)
        self.bridge.load_complete.connect(self.on_load_complete)
        self.bridge.object_clicked.connect(self.on_object_clicked)
        
        layout.addWidget(self.web_view)
        
    def on_load_finished(self, success):
        """页面加载完成回调"""
        if success:
            # 页面加载完成后，再次确保CORS设置生效
            settings = self.web_view.settings()
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent, True)
    
    def load_cesium(self):
        """加载Cesium HTML页面"""
        if not os.path.exists(self.html_path):
            QMessageBox.critical(self, "错误", f"找不到Cesium HTML文件: {self.html_path}")
            return
            
        # 加载本地HTML文件
        url = QUrl.fromLocalFile(os.path.abspath(self.html_path))
        self.web_view.load(url)
        
    def send_message(self, message: Dict[str, Any]):
        """
        发送消息到JavaScript
        
        Args:
            message: 消息字典
        """
        message_json = json.dumps(message, ensure_ascii=False)
        # 转义单引号，避免JavaScript错误
        message_json_escaped = message_json.replace("'", "\\'")
        js_code = f"if (typeof receiveMessageFromPython !== 'undefined') {{ receiveMessageFromPython('{message_json_escaped}'); }} else {{ console.log('[Cesium JS Info] receiveMessageFromPython未定义，消息已排队'); }}"
        self.web_view.page().runJavaScript(js_code)
        
    def load_point_cloud(self, points: list, point_size: int = 2):
        """
        加载点云数据
        
        Args:
            points: 点云数据列表，每个点包含 {longitude, latitude, height, r, g, b, a}
            point_size: 点的大小
        """
        message = {
            'type': 'loadPointCloud',
            'data': {
                'points': points,
                'pointSize': point_size
            }
        }
        self.send_message(message)
        
    def load_3d_tiles(self, url: str, fly_to: bool = True, 
                      maximum_screen_space_error: int = 16):
        """
        加载3D Tiles
        
        Args:
            url: 3D Tiles的URL或本地路径
            fly_to: 是否飞到模型位置
            maximum_screen_space_error: 最大屏幕空间误差
        """
        message = {
            'type': 'load3DTiles',
            'data': {
                'url': url,
                'flyTo': fly_to,
                'maximumScreenSpaceError': maximum_screen_space_error
            }
        }
        self.send_message(message)
    
    def load_3dgs(self, splat_url: str, longitude: float, latitude: float, 
                  altitude: float = 0.0, scale: float = 1.0,
                  rotation_x: float = 0.0, rotation_y: float = 0.0, 
                  rotation_z: float = 0.0, fly_to: bool = True):
        """
        加载3D Gaussian Splatting模型
        
        Args:
            splat_url: .splat文件的URL或本地路径（file://协议）
            longitude: 经度
            latitude: 纬度
            altitude: 海拔高度
            scale: 缩放比例
            rotation_x: X轴旋转（弧度）
            rotation_y: Y轴旋转（弧度）
            rotation_z: Z轴旋转（弧度）
            fly_to: 是否飞到模型位置
        """
        message = {
            'type': 'load3DGS',
            'data': {
                'url': splat_url,
                'location': {
                    'lon': float(longitude),
                    'lat': float(latitude),
                    'height': float(altitude)
                },
                'rotation': {
                    'x': float(rotation_x),
                    'y': float(rotation_y),
                    'z': float(rotation_z)
                },
                'scale': float(scale),
                'flyTo': fly_to
            }
        }
        self.send_message(message)
        
    def clear_all(self):
        """清除所有模型"""
        message = {'type': 'clearAll'}
        self.send_message(message)
        
    def set_camera(self, longitude: float, latitude: float, height: float,
                   heading: float = 0.0, pitch: float = -90.0, roll: float = 0.0):
        """
        设置相机位置
        
        Args:
            longitude: 经度
            latitude: 纬度
            height: 高度
            heading: 航向角
            pitch: 俯仰角
            roll: 翻滚角
        """
        message = {
            'type': 'setCamera',
            'data': {
                'longitude': longitude,
                'latitude': latitude,
                'height': height,
                'heading': heading,
                'pitch': pitch,
                'roll': roll
            }
        }
        self.send_message(message)
        
    def load_segment(self, splat_url: str, longitude: float, latitude: float, 
                     segment_id: int, altitude: float = 0.0, scale: float = 1.0,
                     rotation_x: float = 0.0, rotation_y: float = 0.0, 
                     rotation_z: float = 0.0, fly_to: bool = True):
        """
        加载分割后的3D Gaussian Splatting模型
        
        Args:
            splat_url: .splat文件的URL或本地路径（file://协议）
            longitude: 经度
            latitude: 纬度
            segment_id: 分割ID
            altitude: 海拔高度
            scale: 缩放比例
            rotation_x: X轴旋转（弧度）
            rotation_y: Y轴旋转（弧度）
            rotation_z: Z轴旋转（弧度）
            fly_to: 是否飞到模型位置
        """
        message = {
            'type': 'loadSegment',
            'data': {
                'url': splat_url,
                'location': {
                    'lon': float(longitude),
                    'lat': float(latitude),
                    'height': float(altitude)
                },
                'rotation': {
                    'x': float(rotation_x),
                    'y': float(rotation_y),
                    'z': float(rotation_z)
                },
                'scale': float(scale),
                'flyTo': fly_to,
                'segmentId': segment_id
            }
        }
        self.send_message(message)
        
    def highlight_segment(self, segment_id: int):
        """
        高亮显示分割
        
        Args:
            segment_id: 分割ID
        """
        message = {
            'type': 'highlightSegment',
            'data': {
                'segmentId': segment_id
            }
        }
        self.send_message(message)
        
    def remove_model(self, model_id: str):
        """
        移除指定模型
        
        Args:
            model_id: 模型ID
        """
        message = {
            'type': 'removeModel',
            'data': {
                'modelId': model_id
            }
        }
        self.send_message(message)
        
    def get_model_info(self, model_id: str = None):
        """
        获取模型信息
        
        Args:
            model_id: 模型ID，如果为None则获取所有模型信息
        """
        message = {
            'type': 'getModelInfo',
            'data': {
                'modelId': model_id
            }
        }
        self.send_message(message)
        
    def on_viewer_ready(self):
        """Cesium viewer准备就绪回调"""
        print("[CesiumWidget] Viewer已准备就绪")
        self.viewer_ready.emit()
        
    def on_load_complete(self, success: bool, error: str):
        """
        加载完成回调
        
        Args:
            success: 是否成功
            error: 错误消息（如果有）
        """
        if success:
            print("[CesiumWidget] 模型加载成功")
        else:
            print(f"[CesiumWidget] 模型加载失败: {error}")
        self.load_complete.emit(success, error)
        
    def on_object_clicked(self, data: dict):
        """
        对象点击回调
        
        Args:
            data: 点击数据
        """
        position = data.get('position', {})
        print(f"[CesiumWidget] 点击位置: "
              f"lon={position.get('longitude', 0):.6f}, "
              f"lat={position.get('latitude', 0):.6f}, "
              f"height={position.get('height', 0):.2f}")
        self.object_clicked.emit(data)


class CesiumPanel(QWidget):
    """Cesium面板，包含Cesium viewer和控制按钮"""
    
    def __init__(self, parent=None):
        """
        初始化Cesium面板
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        self.cesium_widget = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建Cesium widget
        self.cesium_widget = CesiumWidget()
        layout.addWidget(self.cesium_widget)
        
        # 连接信号
        self.cesium_widget.viewer_ready.connect(self.on_viewer_ready)
        self.cesium_widget.load_complete.connect(self.on_load_complete)
        self.cesium_widget.object_clicked.connect(self.on_object_clicked)
        
    def on_viewer_ready(self):
        """Viewer准备就绪"""
        print("[CesiumPanel] Cesium viewer已准备就绪")
        
    def on_load_complete(self, success: bool, error: str):
        """加载完成"""
        if not success:
            QMessageBox.warning(self, "加载失败", f"模型加载失败: {error}")
            
    def on_object_clicked(self, data: dict):
        """对象被点击"""
        # 可以在这里实现自定义的点击处理逻辑
        pass
        
    def get_cesium_widget(self) -> CesiumWidget:
        """
        获取Cesium widget
        
        Returns:
            CesiumWidget实例
        """
        return self.cesium_widget


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 创建Cesium面板
    panel = CesiumPanel()
    panel.setWindowTitle("GSSE Cesium Viewer")
    panel.setGeometry(100, 100, 1200, 800)
    panel.show()
    
    # 测试加载点云（viewer准备就绪后）
    def test_load():
        print("测试加载点云...")
        # 生成测试点云
        test_points = []
        for i in range(100):
            test_points.append({
                'longitude': 116.3974 + (i % 10) * 0.001,
                'latitude': 39.9088 + (i // 10) * 0.001,
                'height': 10.0,
                'r': 255,
                'g': 0,
                'b': 0,
                'a': 1.0
            })
        panel.cesium_widget.load_point_cloud(test_points)
    
    # 连接信号测试
    panel.cesium_widget.viewer_ready.connect(test_load)
    
    sys.exit(app.exec_())

