#!/usr/bin/env python3
"""
临时HTTP服务器
用于为Cesium提供本地文件访问，解决file://协议限制问题
"""
import os
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote
import tempfile
import shutil
from typing import Optional, Dict, Any
class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """支持CORS的HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        # 设置服务目录
        self.directory = kwargs.pop('directory', tempfile.gettempdir())
        super().__init__(*args, directory=self.directory, **kwargs)
    
    def end_headers(self):
        """添加CORS头"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def guess_type(self, path):
        """重写guess_type以支持PLY文件"""
        base, ext = os.path.splitext(path)
        if ext.lower() == '.ply':
            return 'application/octet-stream'
        elif ext.lower() == '.splat':
            return 'application/octet-stream'
        return super().guess_type(path)
    
    def do_OPTIONS(self):
        """处理OPTIONS请求（CORS预检）"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """重写日志方法，减少输出"""
        # 只记录错误
        if args and len(args) > 1 and isinstance(args[1], str) and args[1].startswith('4'):
            print(f"[HTTP Server] {format % args}")
class TempHTTPServer:
    """临时HTTP服务器管理器"""
    
    def __init__(self, port: int = 0, base_dir: Optional[str] = None):
        """
        初始化临时HTTP服务器
        
        Args:
            port: 端口号，0表示自动分配
            base_dir: 服务根目录，None表示使用临时目录
        """
        self.port = port
        self.base_dir = base_dir or tempfile.mkdtemp(prefix='gsse_http_')
        self.server = None
        self.server_thread = None
        self.running = False
        self.actual_port = None
        
        # 确保服务目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        
        print(f"[HTTP Server] 服务目录: {self.base_dir}")
    
    def start(self) -> bool:
        """
        启动HTTP服务器
        
        Returns:
            是否启动成功
        """
        if self.running:
            print("[HTTP Server] 服务器已在运行")
            return True
        
        try:
            # 创建服务器
            handler = lambda *args, **kwargs: CORSHTTPRequestHandler(
                *args, directory=self.base_dir, **kwargs
            )
            
            self.server = HTTPServer(('localhost', self.port), handler)
            self.actual_port = self.server.server_address[1]
            
            # 在后台线程中运行服务器
            self.server_thread = threading.Thread(
                target=self._run_server, 
                daemon=True,
                name="TempHTTPServer"
            )
            self.server_thread.start()
            
            # 等待服务器启动
            time.sleep(0.1)
            
            self.running = True
            print(f"[HTTP Server] 服务器已启动: http://localhost:{self.actual_port}")
            return True
            
        except Exception as e:
            print(f"[HTTP Server] 启动失败: {e}")
            return False
    
    def stop(self):
        """停止HTTP服务器"""
        if not self.running:
            return
        
        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2)
            
            self.running = False
            print("[HTTP Server] 服务器已停止")
            
        except Exception as e:
            print(f"[HTTP Server] 停止时出错: {e}")
        finally:
            self.server = None
            self.server_thread = None
    
    def _run_server(self):
        """在后台运行服务器"""
        try:
            self.server.serve_forever()
        except Exception as e:
            if self.running:  # 只有在应该运行时才报告错误
                print(f"[HTTP Server] 运行时出错: {e}")
    
    def add_file(self, file_path: str, url_path: Optional[str] = None) -> str:
        """
        添加文件到服务器
        
        Args:
            file_path: 本地文件路径
            url_path: URL路径，None表示使用文件名
            
        Returns:
            文件的HTTP URL
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not self.running:
            raise RuntimeError("服务器未运行")
        
        # 确定URL路径
        if url_path is None:
            url_path = os.path.basename(file_path)
        
        # 确保URL路径不以/开头
        url_path = url_path.lstrip('/')
        
        # 目标文件路径
        target_path = os.path.join(self.base_dir, url_path)
        
        # 创建目录（如果需要）
        target_dir = os.path.dirname(target_path)
        if target_dir != self.base_dir:
            os.makedirs(target_dir, exist_ok=True)
        
        # 复制文件
        shutil.copy2(file_path, target_path)
        
        # 返回HTTP URL
        http_url = f"http://localhost:{self.actual_port}/{url_path}"
        print(f"[HTTP Server] 文件已添加: {file_path} -> {http_url}")
        return http_url
    
    def add_file_content(self, content: bytes, url_path: str) -> str:
        """
        添加文件内容到服务器
        
        Args:
            content: 文件内容
            url_path: URL路径
            
        Returns:
            文件的HTTP URL
        """
        if not self.running:
            raise RuntimeError("服务器未运行")
        
        # 确保URL路径不以/开头
        url_path = url_path.lstrip('/')
        
        # 目标文件路径
        target_path = os.path.join(self.base_dir, url_path)
        
        # 创建目录（如果需要）
        target_dir = os.path.dirname(target_path)
        if target_dir != self.base_dir:
            os.makedirs(target_dir, exist_ok=True)
        
        # 写入文件
        with open(target_path, 'wb') as f:
            f.write(content)
        
        # 返回HTTP URL
        http_url = f"http://localhost:{self.actual_port}/{url_path}"
        print(f"[HTTP Server] 内容已添加: {http_url}")
        return http_url
    
    def remove_file(self, url_path: str):
        """
        从服务器移除文件
        
        Args:
            url_path: URL路径
        """
        url_path = url_path.lstrip('/')
        target_path = os.path.join(self.base_dir, url_path)
        
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
                print(f"[HTTP Server] 文件已移除: {url_path}")
        except Exception as e:
            print(f"[HTTP Server] 移除文件失败: {e}")
    
    def clear_files(self):
        """清除所有文件"""
        try:
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("[HTTP Server] 所有文件已清除")
        except Exception as e:
            print(f"[HTTP Server] 清除文件失败: {e}")
    
    def get_url(self, url_path: str) -> str:
        """
        获取文件的HTTP URL
        
        Args:
            url_path: URL路径
            
        Returns:
            完整的HTTP URL
        """
        if not self.running:
            raise RuntimeError("服务器未运行")
        
        url_path = url_path.lstrip('/')
        return f"http://localhost:{self.actual_port}/{url_path}"
    
    def is_running(self) -> bool:
        """检查服务器是否运行"""
        return self.running and self.server is not None
    
    def get_port(self) -> Optional[int]:
        """获取实际端口号"""
        return self.actual_port
    
    def get_base_url(self) -> Optional[str]:
        """获取基础URL"""
        if self.running:
            return f"http://localhost:{self.actual_port}"
        return None
    
    def __del__(self):
        """析构函数，确保服务器停止"""
        self.stop()
        
        # 清理临时目录
        try:
            if os.path.exists(self.base_dir) and self.base_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(self.base_dir)
        except Exception:
            pass  # 忽略清理错误
# 全局服务器实例
_global_server: Optional[TempHTTPServer] = None
def get_global_server() -> TempHTTPServer:
    """获取全局HTTP服务器实例"""
    global _global_server
    if _global_server is None:
        _global_server = TempHTTPServer()
    return _global_server
def start_global_server() -> bool:
    """启动全局HTTP服务器"""
    server = get_global_server()
    return server.start()
def stop_global_server():
    """停止全局HTTP服务器"""
    global _global_server
    if _global_server:
        _global_server.stop()
        _global_server = None
def add_file_to_server(file_path: str, url_path: Optional[str] = None) -> str:
    """添加文件到全局服务器"""
    server = get_global_server()
    if not server.is_running():
        server.start()
    return server.add_file(file_path, url_path)
if __name__ == '__main__':
    # 测试代码
    print("测试临时HTTP服务器...")
    
    server = TempHTTPServer()
    
    try:
        # 启动服务器
        if server.start():
            print(f"服务器运行在: {server.get_base_url()}")
            
            # 创建测试文件
            test_file = os.path.join(server.base_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Hello, World!')
            
            print(f"测试文件URL: {server.get_url('test.txt')}")
            
            # 等待用户输入
            input("按Enter键停止服务器...")
        
    finally:
        server.stop()