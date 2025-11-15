#!/usr/bin/env python3
"""
COLMAP处理模块
提供COLMAP自动重建功能，支持多种参数配置
包含视频处理功能，可将视频自动转换为图像序列
"""

import os
import subprocess
import threading
import time
import cv2
from typing import Optional, Callable, Tuple


class COLMAPProcessor:
    """COLMAP处理器类"""
    
    def __init__(self):
        """初始化COLMAP处理器"""
        # 查找COLMAP可执行文件
        self.colmap_path = self._find_colmap_executable()
        self.is_processing = False
        self.progress_callback = None
        self.current_process = None
        
    def _find_colmap_executable(self) -> str:
        """查找COLMAP可执行文件"""
        possible_paths = [
            "/usr/local/bin/colmap",
            "/usr/bin/colmap",
            "colmap"  # 如果在PATH中
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        # 如果没找到，返回默认路径
        return "/usr/local/bin/colmap"
    
    def set_progress_callback(self, callback: Callable[[str, Optional[int]], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str, progress: Optional[int] = None):
        """更新进度"""
        if self.progress_callback and self.is_processing:
            self.progress_callback(message, progress)
    
    def _run_command(self, cmd: list) -> Tuple[int, str, str]:
        """运行命令并返回结果"""
        try:
            self._update_progress(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self._update_progress("命令执行超时", 0)
            return -1, "", "命令执行超时"
        except Exception as e:
            self._update_progress(f"命令执行出错: {str(e)}", 0)
            return -1, "", str(e)
    
    def stop_processing(self):
        """停止处理"""
        self.is_processing = False
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process = None
            except:
                pass
    
    def extract_frames_from_video(self, 
                                  video_path: str, 
                                  output_dir: str,
                                  frame_rate: int = 1,
                                  quality: int = 95,
                                  max_frames: Optional[int] = None,
                                  resize_width: Optional[int] = None) -> Tuple[bool, str, int]:
        """
        从视频中提取图像帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出图像目录
            frame_rate: 每秒提取帧数 (默认1帧/秒)
            quality: JPEG质量 (1-100, 默认95)
            max_frames: 最大提取帧数 (None表示无限制)
            resize_width: 调整图像宽度 (None表示保持原始尺寸)
            
        Returns:
            Tuple[bool, str, int]: (是否成功, 消息, 提取的帧数)
        """
        if not os.path.exists(video_path):
            return False, f"视频文件不存在: {video_path}", 0
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self._update_progress("正在打开视频文件...", 0)
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return False, "无法打开视频文件", 0
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self._update_progress(f"视频信息: {fps:.2f}fps, {total_frames}帧, {duration:.2f}秒", 5)
            
            # 计算帧间隔
            frame_interval = int(fps / frame_rate) if frame_rate > 0 else 1
            
            # 提取帧
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 检查是否应该保存这一帧
                if frame_count % frame_interval == 0:
                    # 调整图像大小（如果指定）
                    if resize_width:
                        h, w = frame.shape[:2]
                        resize_height = int(h * resize_width / w)
                        frame = cv2.resize(frame, (resize_width, resize_height), 
                                         interpolation=cv2.INTER_LANCZOS4)
                    
                    # 保存帧
                    output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    saved_count += 1
                    
                    # 更新进度
                    progress = min(95, int((frame_count / total_frames) * 95))
                    self._update_progress(f"已提取 {saved_count} 帧", progress)
                    
                    # 检查是否达到最大帧数
                    if max_frames and saved_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            self._update_progress(f"视频帧提取完成，共提取 {saved_count} 帧", 100)
            return True, f"成功提取 {saved_count} 帧图像", saved_count
            
        except Exception as e:
            return False, f"提取视频帧时出错: {str(e)}", 0
    
    def auto_reconstruction(self, 
                           image_dir: str, 
                           workspace_path: str,
                           camera_model: str = "OPENCV",
                           single_camera: bool = True,
                           quality: str = "High",
                           data_type: str = "Individual images",
                           mapper_type: str = "incremental",
                           num_threads: str = "-1",
                           sparse_model: bool = True,
                           dense_model: bool = True,
                           use_gpu: bool = True,
                           vocab_tree: str = "") -> bool:
        """
        使用COLMAP的自动重建功能
        
        Args:
            image_dir: 图像目录路径
            workspace_path: 工作空间路径
            camera_model: 相机模型类型
            single_camera: 是否使用单相机模式
            quality: 处理质量 (Low, Medium, High, Extreme)
            data_type: 数据类型 (Individual images, Video)
            mapper_type: 映射器类型 (incremental, global)
            num_threads: 线程数 (-1为自动)
            sparse_model: 是否生成稀疏模型
            dense_model: 是否生成密集模型
            use_gpu: 是否使用GPU加速
            vocab_tree: 词汇树文件路径
            
        Returns:
            bool: 是否成功
        """
        if self.is_processing:
            self._update_progress("COLMAP正在处理中，请等待完成", 0)
            return False
            
        self.is_processing = True
        
        try:
            # 创建输出目录
            os.makedirs(workspace_path, exist_ok=True)
            
            self._update_progress("开始COLMAP自动重建...", 0)
            
            # 构建COLMAP命令
            cmd = [
                self.colmap_path, "automatic_reconstructor",
                "--workspace_path", workspace_path,
                "--image_path", image_dir,
                "--camera_model", camera_model,
                "--single_camera", str(single_camera).lower()
            ]
            
            # 添加质量参数
            quality_map = {
                "Low": "low",
                "Medium": "medium", 
                "High": "high",
                "Extreme": "extreme"
            }
            if quality in quality_map:
                cmd.extend(["--quality", quality_map[quality]])
            
            # 添加数据类型参数
            if data_type == "Video":
                cmd.append("--video")
            
            # 添加映射器类型参数
            if mapper_type == "global":
                cmd.append("--global")
            
            # 添加线程数参数
            if num_threads != "-1":
                cmd.extend(["--num_threads", num_threads])
            
            # 添加稀疏模型参数
            if not sparse_model:
                cmd.append("--no_sparse")
            
            # 添加密集模型参数
            if not dense_model:
                cmd.append("--no_dense")
            
            # 添加GPU参数
            if not use_gpu:
                cmd.append("--no_gpu")
            
            # 添加词汇树参数
            if vocab_tree and os.path.exists(vocab_tree):
                cmd.extend(["--vocab_tree_path", vocab_tree])
            
            self._update_progress("执行COLMAP自动重建...", 10)
            
            # 执行命令
            returncode, stdout, stderr = self._run_command(cmd)
            
            if returncode == 0:
                self._update_progress("COLMAP自动重建完成", 90)
                
                # 确保images文件夹存在
                images_dir = os.path.join(workspace_path, "images")
                if not os.path.exists(images_dir):
                    self._update_progress("创建images文件夹...", 95)
                    os.makedirs(images_dir, exist_ok=True)
                    
                    # 复制原始图像到images文件夹
                    import shutil
                    try:
                        for file in os.listdir(image_dir):
                            if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']):
                                src_path = os.path.join(image_dir, file)
                                dst_path = os.path.join(images_dir, file)
                                shutil.copy2(src_path, dst_path)
                        self._update_progress("images文件夹创建完成", 100)
                    except Exception as e:
                        self._update_progress(f"复制图像到images文件夹失败: {str(e)}", 100)
                
                return True
            else:
                self._update_progress(f"COLMAP自动重建失败: {stderr}", 0)
                return False
                
        except Exception as e:
            self._update_progress(f"COLMAP自动重建出错: {str(e)}", 0)
            return False
        finally:
            self.is_processing = False


def validate_video_file(video_path: str) -> Tuple[bool, str]:
    """
    验证视频文件是否有效
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        Tuple[bool, str]: (是否有效, 消息)
    """
    if not os.path.exists(video_path):
        return False, f"视频文件不存在: {video_path}"
    
    if not os.path.isfile(video_path):
        return False, f"路径不是文件: {video_path}"
    
    # 检查支持的视频格式
    supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    file_ext = os.path.splitext(video_path)[1].lower()
    
    if file_ext not in supported_extensions:
        return False, f"不支持的视频格式: {file_ext}，支持的格式: {', '.join(supported_extensions)}"
    
    # 尝试打开视频文件
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "无法打开视频文件"
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        if total_frames < 2:
            return False, f"视频帧数不足，需要至少2帧，当前: {total_frames}"
        
        return True, f"视频信息: {width}x{height}, {fps:.2f}fps, {total_frames}帧, {duration:.2f}秒"
    except Exception as e:
        return False, f"读取视频文件出错: {str(e)}"


def get_video_info(video_path: str) -> dict:
    """
    获取视频文件信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 视频信息字典
    """
    info = {
        'valid': False,
        'fps': 0,
        'total_frames': 0,
        'width': 0,
        'height': 0,
        'duration': 0,
        'error': ''
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            info['error'] = "无法打开视频文件"
            return info
        
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        info['valid'] = True
        
        cap.release()
    except Exception as e:
        info['error'] = str(e)
    
    return info


def validate_image_directory(image_dir: str) -> Tuple[bool, str]:
    """
    验证图像目录是否有效
    
    Args:
        image_dir: 图像目录路径
        
    Returns:
        Tuple[bool, str]: (是否有效, 消息)
    """
    if not os.path.exists(image_dir):
        return False, f"目录不存在: {image_dir}"
    
    if not os.path.isdir(image_dir):
        return False, f"路径不是目录: {image_dir}"
    
    # 检查支持的图像格式
    supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(file)
    
    if len(image_files) < 2:
        return False, f"图像数量不足，需要至少2张图像，当前找到: {len(image_files)}"
    
    return True, f"找到 {len(image_files)} 张图像"


def convert_colmap_to_gaussian_splatting_format(colmap_output_dir: str, 
                                               gs_output_dir: str, 
                                               original_image_dir: str) -> bool:
    """
    将COLMAP输出转换为3D Gaussian Splatting格式
    
    Args:
        colmap_output_dir: COLMAP输出目录
        gs_output_dir: 3DGS输出目录
        original_image_dir: 原始图像目录
        
    Returns:
        bool: 是否成功
    """
    try:
        # 创建输出目录
        os.makedirs(gs_output_dir, exist_ok=True)
        
        # 复制sparse目录
        sparse_src = os.path.join(colmap_output_dir, "sparse")
        sparse_dst = os.path.join(gs_output_dir, "sparse")
        
        if os.path.exists(sparse_src):
            import shutil
            if os.path.exists(sparse_dst):
                shutil.rmtree(sparse_dst)
            shutil.copytree(sparse_src, sparse_dst)
        
        # 复制图像
        images_dst = os.path.join(gs_output_dir, "images")
        
        # 优先使用COLMAP输出的images文件夹
        images_src = os.path.join(colmap_output_dir, "images")
        if os.path.exists(images_src):
            import shutil
            if os.path.exists(images_dst):
                shutil.rmtree(images_dst)
            shutil.copytree(images_src, images_dst)
        else:
            # 如果没有images文件夹，使用去畸变图像
            undistorted_src = os.path.join(colmap_output_dir, "dense", "images")
            if os.path.exists(undistorted_src):
                import shutil
                if os.path.exists(images_dst):
                    shutil.rmtree(images_dst)
                shutil.copytree(undistorted_src, images_dst)
            else:
                # 最后使用原始图像
                import shutil
                if os.path.exists(images_dst):
                    shutil.rmtree(images_dst)
                shutil.copytree(original_image_dir, images_dst)
        
        # 生成cameras.json文件
        cameras_json_path = os.path.join(gs_output_dir, "cameras.json")
        generate_cameras_json(sparse_dst, cameras_json_path)
        
        return True
        
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False


def generate_cameras_json(sparse_dir: str, output_path: str):
    """
    生成cameras.json文件
    
    Args:
        sparse_dir: COLMAP sparse目录
        output_path: 输出JSON文件路径
    """
    try:
        # 这是一个简化的实现
        # 实际应用中需要解析COLMAP的二进制文件
        cameras_data = {
            "id": 0,
            "img_name": "placeholder",
            "width": 1920,
            "height": 1080,
            "position": [0.0, 0.0, 0.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "fy": 1000.0,
            "fx": 1000.0
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(cameras_data, f, indent=2)
            
    except Exception as e:
        print(f"生成cameras.json失败: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    processor = COLMAPProcessor()
    print(f"COLMAP路径: {processor.colmap_path}")
    
    # 测试图像目录验证（使用相对路径）
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, "colmap", "input", "miku")
    is_valid, message = validate_image_directory(test_dir)
    print(f"图像目录验证: {is_valid}, {message}")