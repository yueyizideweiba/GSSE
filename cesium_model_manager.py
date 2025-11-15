#!/usr/bin/env python3
"""
Cesium模型管理器
参照cesium-gaussian-splatting项目实现，用于管理Cesium中的3DGS模型和分割模型
"""

import os
import json
import tempfile
import shutil
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from cesium_widget import CesiumWidget
from gis_converter import GaussianSplattingToGIS, CoordinateTransformer
from ply_to_splat_converter import convert_gaussians_to_splat
from exif_geo_extractor import get_model_center_from_images


class CesiumModelManager:
    """Cesium模型管理器"""
    
    def __init__(self, cesium_widget: CesiumWidget):
        """
        初始化模型管理器
        
        Args:
            cesium_widget: CesiumWidget实例
        """
        self.cesium_widget = cesium_widget
        self.gis_converter = GaussianSplattingToGIS()
        
        # 模型管理
        self.loaded_models = {}  # model_id -> model_info
        self.temp_dirs = []  # 临时目录列表
        
        # 自动定位配置
        self.auto_geo_location = True  # 是否启用自动地理定位
        
    def load_full_model_to_cesium(self, gaussians, longitude: float, latitude: float, 
                                 altitude: float = 0.0, model_id: str = None,
                                 scale: float = 1.0, fly_to: bool = True) -> str:
        """
        加载完整3DGS模型到Cesium
        
        Args:
            gaussians: GaussianModel对象
            longitude: 经度
            latitude: 纬度
            altitude: 海拔高度
            model_id: 模型ID，如果为None则自动生成
            scale: 缩放比例
            fly_to: 是否飞到模型位置
            
        Returns:
            模型ID
        """
        if model_id is None:
            model_id = f"model_{len(self.loaded_models) + 1}"
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix=f'gsse_cesium_{model_id}_')
        self.temp_dirs.append(temp_dir)
        
        # 转换模型为.splat格式
        splat_path = os.path.join(temp_dir, f'{model_id}.splat')
        convert_gaussians_to_splat(gaussians, splat_path)
        
        # 转换为file://URL
        splat_url = f"file://{splat_path}"
        
        # 加载到Cesium
        self.cesium_widget.load_3dgs(
            splat_url, longitude, latitude, altitude, 
            scale=scale, fly_to=fly_to
        )
        
        # 保存模型信息
        self.loaded_models[model_id] = {
            'type': 'full_model',
            'splat_path': splat_path,
            'temp_dir': temp_dir,
            'geo_location': {
                'longitude': longitude,
                'latitude': latitude,
                'altitude': altitude
            },
            'scale': scale,
            'gaussians': gaussians  # 保留引用，用于后续分割
        }
        
        return model_id
    
    def auto_detect_geo_location(self, image_directory: str = None, 
                               gaussians = None) -> Optional[Dict[str, float]]:
        """
        自动检测模型的地理位置
        
        Args:
            image_directory: 包含训练图像的目录
            gaussians: GaussianModel对象（用于获取图像路径）
            
        Returns:
            地理坐标字典，格式为 {'longitude': float, 'latitude': float, 'altitude': float}
            如果无法检测则返回None
        """
        if not self.auto_geo_location:
            return None
        
        # 优先使用提供的图像目录
        if image_directory and os.path.exists(image_directory):
            gps_data = get_model_center_from_images(image_directory)
            if gps_data:
                return {
                    'longitude': gps_data.get('longitude', 0.0),
                    'latitude': gps_data.get('latitude', 0.0),
                    'altitude': gps_data.get('altitude', 0.0)
                }
        
        # 如果gaussians对象有图像路径信息，尝试从中提取
        if gaussians and hasattr(gaussians, 'source_path'):
            source_path = gaussians.source_path
            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    gps_data = get_model_center_from_images(source_path)
                else:
                    # 如果是文件，尝试从父目录查找
                    parent_dir = os.path.dirname(source_path)
                    if os.path.exists(parent_dir):
                        gps_data = get_model_center_from_images(parent_dir)
                
                if gps_data:
                    return {
                        'longitude': gps_data.get('longitude', 0.0),
                        'latitude': gps_data.get('latitude', 0.0),
                        'altitude': gps_data.get('altitude', 0.0)
                    }
        
        # 如果都没有找到GPS数据，返回None
        return None
    
    def load_full_model_auto_geo(self, gaussians, image_directory: str = None,
                               model_id: str = None, scale: float = 1.0, 
                               fly_to: bool = True) -> str:
        """
        自动检测地理位置并加载完整3DGS模型到Cesium
        
        Args:
            gaussians: GaussianModel对象
            image_directory: 包含训练图像的目录
            model_id: 模型ID，如果为None则自动生成
            scale: 缩放比例
            fly_to: 是否飞到模型位置
            
        Returns:
            模型ID
        """
        # 自动检测地理位置
        geo_location = self.auto_detect_geo_location(image_directory, gaussians)
        
        if geo_location:
            print(f"[INFO] 自动检测到模型位置: "
                  f"经度={geo_location['longitude']:.6f}, "
                  f"纬度={geo_location['latitude']:.6f}, "
                  f"海拔={geo_location['altitude']:.2f}m")
            
            return self.load_full_model_to_cesium(
                gaussians, 
                geo_location['longitude'], 
                geo_location['latitude'], 
                geo_location['altitude'],
                model_id=model_id, 
                scale=scale, 
                fly_to=fly_to
            )
        else:
            # 如果无法自动检测，使用默认位置并提示用户
            print("[WARNING] 无法自动检测模型位置，使用默认位置（北京天安门）")
            return self.load_full_model_to_cesium(
                gaussians, 
                116.3974,  # 北京天安门经度
                39.9088,   # 北京天安门纬度
                0.0,       # 海拔
                model_id=model_id, 
                scale=scale, 
                fly_to=fly_to
            )
    
    def load_segment_to_cesium(self, gaussians, mask: list, segment_id: int,
                              longitude: float, latitude: float, 
                              altitude: float = 0.0, scale: float = 1.0,
                              fly_to: bool = True) -> str:
        """
        加载分割后的3DGS模型到Cesium
        
        Args:
            gaussians: GaussianModel对象
            mask: 分割掩码（布尔列表）
            segment_id: 分割ID
            longitude: 经度
            latitude: 纬度
            altitude: 海拔高度
            scale: 缩放比例
            fly_to: 是否飞到模型位置
            
        Returns:
            模型ID
        """
        import numpy as np
        
        model_id = f"segment_{segment_id}"
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix=f'gsse_cesium_segment_{segment_id}_')
        self.temp_dirs.append(temp_dir)
        
        # 转换掩码为numpy数组
        mask_array = np.array(mask, dtype=bool)
        
        # 转换分割模型为.splat格式
        splat_path = os.path.join(temp_dir, f'segment_{segment_id}.splat')
        convert_gaussians_to_splat(gaussians, splat_path, mask_array)
        
        # 转换为file://URL
        splat_url = f"file://{splat_path}"
        
        # 加载到Cesium
        self.cesium_widget.load_segment(
            splat_url, longitude, latitude, segment_id,
            altitude=altitude, scale=scale, fly_to=fly_to
        )
        
        # 保存模型信息
        self.loaded_models[model_id] = {
            'type': 'segment',
            'segment_id': segment_id,
            'splat_path': splat_path,
            'temp_dir': temp_dir,
            'geo_location': {
                'longitude': longitude,
                'latitude': latitude,
                'altitude': altitude
            },
            'scale': scale,
            'mask': mask_array
        }
        
        return model_id
    
    def highlight_segment(self, segment_id: int):
        """
        高亮显示分割模型
        
        Args:
            segment_id: 分割ID
        """
        self.cesium_widget.highlight_segment(segment_id)
    
    def remove_model(self, model_id: str):
        """
        移除指定模型
        
        Args:
            model_id: 模型ID
        """
        if model_id in self.loaded_models:
            # 从Cesium中移除
            self.cesium_widget.remove_model(model_id)
            
            # 清理临时文件
            model_info = self.loaded_models[model_id]
            if os.path.exists(model_info['temp_dir']):
                shutil.rmtree(model_info['temp_dir'])
            
            # 从管理器中移除
            del self.loaded_models[model_id]
    
    def clear_all_models(self):
        """清除所有模型"""
        # 从Cesium中清除
        self.cesium_widget.clear_all()
        
        # 清理所有临时文件
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # 清空管理器
        self.loaded_models.clear()
        self.temp_dirs.clear()
    
    def get_model_info(self, model_id: str = None) -> Dict:
        """
        获取模型信息
        
        Args:
            model_id: 模型ID，如果为None则获取所有模型信息
            
        Returns:
            模型信息字典
        """
        if model_id is None:
            return {
                'total_models': len(self.loaded_models),
                'models': list(self.loaded_models.keys()),
                'model_details': self.loaded_models
            }
        
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        else:
            return {'error': f'模型 {model_id} 不存在'}
    
    def export_model_to_gis(self, model_id: str, output_path: str, 
                           format_type: str = 'point_cloud') -> str:
        """
        导出模型为GIS格式
        
        Args:
            model_id: 模型ID
            output_path: 输出路径
            format_type: 格式类型 ('point_cloud', '3d_tiles')
            
        Returns:
            输出文件路径
        """
        if model_id not in self.loaded_models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model_info = self.loaded_models[model_id]
        
        if format_type == 'point_cloud':
            # 导出为点云JSON格式
            return self.gis_converter.convert_to_point_cloud_json(
                model_info['gaussians'], output_path
            )
        elif format_type == '3d_tiles':
            # 导出为3D Tiles格式
            return self.gis_converter.convert_to_3d_tiles(
                model_info['gaussians'], output_path
            )
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
    
    def set_camera_to_model(self, model_id: str, distance: float = 100.0):
        """
        设置相机到模型位置
        
        Args:
            model_id: 模型ID
            distance: 相机距离（米）
        """
        if model_id not in self.loaded_models:
            raise ValueError(f"模型 {model_id} 不存在")
        
        model_info = self.loaded_models[model_id]
        geo_loc = model_info['geo_location']
        
        # 设置相机位置
        self.cesium_widget.set_camera(
            geo_loc['longitude'],
            geo_loc['latitude'],
            geo_loc['altitude'] + distance,
            pitch=-45,  # 俯视角度
            heading=0    # 正北方向
        )
    
    def __del__(self):
        """析构函数，清理临时文件"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass  # 忽略清理错误


class CesiumSegmentManager:
    """Cesium分割模型管理器（专门用于管理分割结果）"""
    
    def __init__(self, cesium_widget: CesiumWidget):
        """
        初始化分割管理器
        
        Args:
            cesium_widget: CesiumWidget实例
        """
        self.cesium_widget = cesium_widget
        self.model_manager = CesiumModelManager(cesium_widget)
        
        # 分割管理
        self.segments = {}  # segment_id -> segment_info
        self.current_highlighted = None
    
    def load_segments(self, gaussians, segment_masks: Dict[int, list], 
                     longitude: float, latitude: float, 
                     altitude: float = 0.0, spacing: float = 10.0):
        """
        加载多个分割模型到Cesium
        
        Args:
            gaussians: GaussianModel对象
            segment_masks: 分割掩码字典 {segment_id: mask_list}
            longitude: 基准经度
            latitude: 基准纬度
            altitude: 基准海拔
            spacing: 分割模型之间的间距（米）
        """
        segment_ids = list(segment_masks.keys())
        
        for i, (segment_id, mask) in enumerate(segment_masks.items()):
            # 计算分割模型的偏移位置
            offset_x = (i % 3) * spacing  # 每行最多3个
            offset_y = (i // 3) * spacing
            
            # 转换为地理坐标偏移
            lat_offset = offset_y / 111000.0  # 1度纬度约111km
            lon_offset = offset_x / (111000.0 * abs(np.cos(np.radians(latitude))))
            
            segment_lon = longitude + lon_offset
            segment_lat = latitude + lat_offset
            
            # 加载分割模型
            model_id = self.model_manager.load_segment_to_cesium(
                gaussians, mask, segment_id,
                segment_lon, segment_lat, altitude,
                scale=1.0, fly_to=False  # 不自动飞行
            )
            
            # 保存分割信息
            self.segments[segment_id] = {
                'model_id': model_id,
                'geo_location': {
                    'longitude': segment_lon,
                    'latitude': segment_lat,
                    'altitude': altitude
                },
                'mask': mask
            }
    
    def highlight_segment(self, segment_id: int, color: str = '#ff0000'):
        """
        高亮显示指定分割
        
        Args:
            segment_id: 分割ID
            color: 高亮颜色
        """
        if segment_id in self.segments:
            # 取消之前的高亮
            if self.current_highlighted:
                self._remove_highlight(self.current_highlighted)
            
            # 应用高亮
            self._apply_highlight(segment_id, color)
            self.current_highlighted = segment_id
    
    def _apply_highlight(self, segment_id: int, color: str):
        """应用高亮效果（待实现）"""
        # 这里可以实现更复杂的高亮效果
        # 例如：改变模型颜色、添加边框等
        pass
    
    def _remove_highlight(self, segment_id: int):
        """移除高亮效果（待实现）"""
        # 恢复原始外观
        pass
    
    def export_segments_report(self, output_path: str) -> str:
        """
        导出分割报告
        
        Args:
            output_path: 输出路径
            
        Returns:
            报告文件路径
        """
        report = {
            'total_segments': len(self.segments),
            'segments': {}
        }
        
        for segment_id, segment_info in self.segments.items():
            mask = segment_info['mask']
            report['segments'][segment_id] = {
                'point_count': np.sum(mask) if mask is not None else 0,
                'geo_location': segment_info['geo_location'],
                'model_id': segment_info['model_id']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path