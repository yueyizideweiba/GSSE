#!/usr/bin/env python3
"""
GIS数据转换器
用于将3DGS模型转换为Cesium可加载的格式
"""

import os
import json
import numpy as np
from typing import Optional, Tuple, Dict, List
import torch
from pathlib import Path


class CoordinateTransformer:
    """坐标转换器"""
    
    def __init__(self):
        """初始化坐标转换器"""
        self.origin_lon = 0.0  # 原点经度
        self.origin_lat = 0.0  # 原点纬度
        self.origin_alt = 0.0  # 原点高度
        self.scale = 1.0       # 缩放比例
        self.rotation = 0.0    # 旋转角度
        
    def set_origin(self, longitude: float, latitude: float, altitude: float = 0.0):
        """
        设置坐标系原点
        
        Args:
            longitude: 经度
            latitude: 纬度
            altitude: 海拔高度
        """
        self.origin_lon = longitude
        self.origin_lat = latitude
        self.origin_alt = altitude
        
    def set_scale(self, scale: float):
        """
        设置缩放比例
        
        Args:
            scale: 缩放比例（本地坐标单位对应的米数）
        """
        self.scale = scale
        
    def set_rotation(self, rotation: float):
        """
        设置旋转角度
        
        Args:
            rotation: 旋转角度（度）
        """
        self.rotation = rotation
        
    def local_to_geo(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        将本地坐标转换为地理坐标
        
        Args:
            x, y, z: 本地坐标
            
        Returns:
            (longitude, latitude, altitude): 地理坐标
        """
        # 应用缩放
        x_m = x * self.scale
        y_m = y * self.scale
        z_m = z * self.scale
        
        # 应用旋转（绕z轴）
        angle_rad = np.radians(self.rotation)
        x_rot = x_m * np.cos(angle_rad) - y_m * np.sin(angle_rad)
        y_rot = x_m * np.sin(angle_rad) + y_m * np.cos(angle_rad)
        
        # 转换为地理坐标（简化版本，适用于小范围）
        # 1度纬度约等于111km，1度经度取决于纬度
        lat_offset = y_rot / 111000.0
        lon_offset = x_rot / (111000.0 * np.cos(np.radians(self.origin_lat)))
        
        longitude = self.origin_lon + lon_offset
        latitude = self.origin_lat + lat_offset
        altitude = self.origin_alt + z_m
        
        return longitude, latitude, altitude
        
    def geo_to_local(self, longitude: float, latitude: float, altitude: float) -> Tuple[float, float, float]:
        """
        将地理坐标转换为本地坐标
        
        Args:
            longitude, latitude, altitude: 地理坐标
            
        Returns:
            (x, y, z): 本地坐标
        """
        # 计算偏移量（米）
        lon_offset = (longitude - self.origin_lon) * 111000.0 * np.cos(np.radians(self.origin_lat))
        lat_offset = (latitude - self.origin_lat) * 111000.0
        alt_offset = altitude - self.origin_alt
        
        # 应用反向旋转
        angle_rad = np.radians(-self.rotation)
        x_m = lon_offset * np.cos(angle_rad) - lat_offset * np.sin(angle_rad)
        y_m = lon_offset * np.sin(angle_rad) + lat_offset * np.cos(angle_rad)
        
        # 应用反向缩放
        x = x_m / self.scale
        y = y_m / self.scale
        z = alt_offset / self.scale
        
        return x, y, z


class GaussianSplattingToGIS:
    """3D Gaussian Splatting到GIS格式转换器"""
    
    def __init__(self, transformer: Optional[CoordinateTransformer] = None):
        """
        初始化转换器
        
        Args:
            transformer: 坐标转换器，如果为None则创建默认的
        """
        self.transformer = transformer if transformer else CoordinateTransformer()
        
    def extract_points_from_gaussians(self, gaussians, mask: Optional[np.ndarray] = None) -> Dict:
        """
        从Gaussian Splatting模型提取点云数据
        
        Args:
            gaussians: GaussianModel对象
            mask: 可选掩码，用于提取特定分割的点
            
        Returns:
            包含点云数据的字典
        """
        # 获取点的位置
        positions = gaussians.get_xyz.detach().cpu().numpy()
        
        # 获取颜色（SH系数转RGB）
        features = gaussians.get_features.detach().cpu().numpy()
        
        # 获取透明度
        opacities = gaussians.get_opacity.detach().cpu().numpy()
        
        # 获取缩放和旋转（可选）
        scales = gaussians.get_scaling.detach().cpu().numpy()
        rotations = gaussians.get_rotation.detach().cpu().numpy()
        
        # 应用掩码（如果提供）
        if mask is not None:
            if len(mask) != len(positions):
                raise ValueError(f"掩码长度({len(mask)})与点数({len(positions)})不匹配")
            
            positions = positions[mask]
            features = features[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            rotations = rotations[mask]
        
        # 将SH特征转换为RGB
        C0 = 0.28209479177387814
        rgb = features[:, 0:3] * C0 + 0.5
        rgb = np.clip(rgb, 0, 1) * 255
        
        return {
            'positions': positions,
            'colors': rgb.astype(np.uint8),
            'opacities': opacities,
            'scales': scales,
            'rotations': rotations,
            'num_points': len(positions),
            'has_mask': mask is not None
        }
        
    def convert_to_point_cloud_json(self, gaussians, output_path: str, 
                                    sample_rate: float = 1.0,
                                    opacity_threshold: float = 0.1) -> str:
        """
        将Gaussian Splatting模型转换为点云JSON格式
        
        Args:
            gaussians: GaussianModel对象
            output_path: 输出文件路径
            sample_rate: 采样率（0-1），用于减少点数
            opacity_threshold: 透明度阈值，低于此值的点将被过滤
            
        Returns:
            输出文件路径
        """
        # 提取点云数据
        data = self.extract_points_from_gaussians(gaussians)
        
        positions = data['positions']
        colors = data['colors']
        opacities = data['opacities'].flatten()
        
        # 过滤低透明度的点
        mask = opacities > opacity_threshold
        positions = positions[mask]
        colors = colors[mask]
        opacities = opacities[mask]
        
        # 采样
        if sample_rate < 1.0:
            num_points = int(len(positions) * sample_rate)
            indices = np.random.choice(len(positions), num_points, replace=False)
            positions = positions[indices]
            colors = colors[indices]
            opacities = opacities[indices]
        
        # 转换坐标
        points_geo = []
        for i in range(len(positions)):
            # 确保所有值都是Python标量
            x = float(positions[i, 0])
            y = float(positions[i, 1])
            z = float(positions[i, 2])
            
            lon, lat, alt = self.transformer.local_to_geo(x, y, z)
            
            points_geo.append({
                'longitude': float(lon),
                'latitude': float(lat),
                'height': float(alt),
                'r': int(colors[i, 0]),
                'g': int(colors[i, 1]),
                'b': int(colors[i, 2]),
                'a': float(opacities[i])
            })
        
        # 构建输出数据
        output_data = {
            'type': 'PointCloud',
            'version': '1.0',
            'origin': {
                'longitude': self.transformer.origin_lon,
                'latitude': self.transformer.origin_lat,
                'altitude': self.transformer.origin_alt
            },
            'pointCount': len(points_geo),
            'points': points_geo
        }
        
        # 保存JSON
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f)
            
        print(f"点云已导出到: {output_path}")
        print(f"点数: {len(points_geo)}")
        
        return output_path
        
    def convert_to_3dtiles(self, gaussians, output_dir: str,
                          tile_size: int = 10000,
                          sample_rate: float = 1.0,
                          opacity_threshold: float = 0.1) -> str:
        """
        将Gaussian Splatting模型转换为3D Tiles格式
        
        Args:
            gaussians: GaussianModel对象
            output_dir: 输出目录
            tile_size: 每个瓦片的最大点数
            sample_rate: 采样率（0-1）
            opacity_threshold: 透明度阈值
            
        Returns:
            tileset.json的路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取点云数据
        data = self.extract_points_from_gaussians(gaussians)
        
        positions = data['positions']
        colors = data['colors']
        opacities = data['opacities'].flatten()
        
        # 过滤和采样
        mask = opacities > opacity_threshold
        positions = positions[mask]
        colors = colors[mask]
        opacities = opacities[mask]
        
        if sample_rate < 1.0:
            num_points = int(len(positions) * sample_rate)
            indices = np.random.choice(len(positions), num_points, replace=False)
            positions = positions[indices]
            colors = colors[indices]
            opacities = opacities[indices]
        
        # 计算边界框
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        center_local = (min_pos + max_pos) / 2
        
        # 转换中心点到地理坐标（确保使用Python标量）
        center_lon, center_lat, center_alt = self.transformer.local_to_geo(
            float(center_local[0]), float(center_local[1]), float(center_local[2])
        )
        
        # 计算边界半径
        half_size = (max_pos - min_pos) / 2
        radius = float(np.linalg.norm(half_size) * self.transformer.scale)
        
        # 创建tileset.json
        tileset = {
            "asset": {
                "version": "1.0",
                "generator": "GSSE GIS Converter"
            },
            "geometricError": radius,
            "root": {
                "boundingVolume": {
                    "sphere": [center_lon, center_lat, center_alt, radius]
                },
                "geometricError": radius / 2,
                "refine": "ADD",
                "content": {
                    "uri": "content.pnts"
                }
            }
        }
        
        tileset_path = os.path.join(output_dir, "tileset.json")
        with open(tileset_path, 'w') as f:
            json.dump(tileset, f, indent=2)
        
        # 创建PNTS文件（Point Cloud格式）
        # 这是一个简化版本，完整实现需要二进制编码
        pnts_data = {
            "points": []
        }
        
        for i in range(len(positions)):
            # 确保所有值都是Python标量
            x = float(positions[i, 0])
            y = float(positions[i, 1])
            z = float(positions[i, 2])
            
            lon, lat, alt = self.transformer.local_to_geo(x, y, z)
            
            pnts_data["points"].append({
                "position": [float(lon), float(lat), float(alt)],
                "color": [int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])],
                "alpha": float(opacities[i])
            })
        
        pnts_path = os.path.join(output_dir, "content.pnts.json")
        with open(pnts_path, 'w') as f:
            json.dump(pnts_data, f)
        
        print(f"3D Tiles已导出到: {output_dir}")
        print(f"Tileset路径: {tileset_path}")
        
        return tileset_path


class GeoDataConfig:
    """地理数据配置管理"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # 默认配置
            return {
                "origin": {
                    "longitude": 116.3974,  # 北京天安门
                    "latitude": 39.9088,
                    "altitude": 0.0
                },
                "scale": 1.0,
                "rotation": 0.0,
                "coordinate_system": "WGS84"
            }
    
    def save_config(self):
        """保存配置文件"""
        if self.config_path:
            os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else '.', exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
    def get_transformer(self) -> CoordinateTransformer:
        """获取配置的坐标转换器"""
        transformer = CoordinateTransformer()
        origin = self.config.get('origin', {})
        transformer.set_origin(
            origin.get('longitude', 0.0),
            origin.get('latitude', 0.0),
            origin.get('altitude', 0.0)
        )
        transformer.set_scale(self.config.get('scale', 1.0))
        transformer.set_rotation(self.config.get('rotation', 0.0))
        return transformer
        
    def update_origin(self, longitude: float, latitude: float, altitude: float = 0.0):
        """更新原点"""
        self.config['origin'] = {
            'longitude': longitude,
            'latitude': latitude,
            'altitude': altitude
        }
        self.save_config()
        
    def update_scale(self, scale: float):
        """更新缩放"""
        self.config['scale'] = scale
        self.save_config()
        
    def update_rotation(self, rotation: float):
        """更新旋转"""
        self.config['rotation'] = rotation
        self.save_config()


# 一些预设的地理坐标配置
PRESET_LOCATIONS = {
    "北京天安门": {"longitude": 116.3974, "latitude": 39.9088, "altitude": 0.0},
    "上海外滩": {"longitude": 121.4897, "latitude": 31.2397, "altitude": 0.0},
    "深圳市民中心": {"longitude": 114.0579, "latitude": 22.5455, "altitude": 0.0},
    "纽约时代广场": {"longitude": -73.9855, "latitude": 40.7580, "altitude": 0.0},
    "伦敦大本钟": {"longitude": -0.1246, "latitude": 51.5007, "altitude": 0.0},
    "巴黎埃菲尔铁塔": {"longitude": 2.2945, "latitude": 48.8584, "altitude": 0.0},
}


if __name__ == '__main__':
    # 测试代码
    print("GIS转换器模块加载成功")
    
    # 测试坐标转换
    transformer = CoordinateTransformer()
    transformer.set_origin(116.3974, 39.9088, 0.0)
    transformer.set_scale(1.0)
    
    # 测试本地到地理坐标转换
    lon, lat, alt = transformer.local_to_geo(100, 100, 10)
    print(f"本地坐标(100, 100, 10) -> 地理坐标({lon:.6f}, {lat:.6f}, {alt:.2f})")
    
    # 测试地理到本地坐标转换
    x, y, z = transformer.geo_to_local(lon, lat, alt)
    print(f"地理坐标({lon:.6f}, {lat:.6f}, {alt:.2f}) -> 本地坐标({x:.2f}, {y:.2f}, {z:.2f})")

