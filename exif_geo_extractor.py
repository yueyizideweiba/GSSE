#!/usr/bin/env python3
"""
EXIF地理数据提取工具
用于从图像文件中提取GPS坐标信息
"""

import os
import json
from typing import Optional, Dict, List, Tuple
from PIL import Image, ExifTags
import numpy as np


def dms_to_decimal(degrees: Tuple[int, int, int], ref: str) -> float:
    """
    将度分秒格式转换为十进制格式
    
    Args:
        degrees: (度, 分, 秒) 元组
        ref: 参考方向 ('N', 'S', 'E', 'W')
        
    Returns:
        十进制坐标值
    """
    if not degrees:
        return 0.0
    
    # 确保是浮点数
    d, m, s = map(float, degrees)
    
    # 计算十进制值
    decimal = d + m/60.0 + s/3600.0
    
    # 根据参考方向调整符号
    if ref in ['S', 'W']:
        decimal = -decimal
    
    return decimal


def extract_gps_from_exif(exif_data: Dict) -> Optional[Dict[str, float]]:
    """
    从EXIF数据中提取GPS坐标
    
    Args:
        exif_data: EXIF数据字典
        
    Returns:
        包含经纬度的字典，格式为 {'latitude': float, 'longitude': float, 'altitude': float}
        如果没有GPS数据则返回None
    """
    if not exif_data:
        return None
    
    # 查找GPSInfo标签（GPS数据通常嵌套在GPSInfo中）
    gps_info_dict = None
    for tag_id, value in exif_data.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        if tag == 'GPSInfo':
            gps_info_dict = value
            break
    
    # 如果没有找到GPSInfo，尝试直接查找GPS标签
    if gps_info_dict is None:
        gps_info_dict = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag.startswith('GPS'):
                gps_info_dict[tag] = value
    
    if not gps_info_dict:
        return None
    
    gps_info = {}
    
    # 提取GPS标签的数值ID
    gps_tag_ids = {}
    for tag_id, tag_name in ExifTags.TAGS.items():
        if tag_name.startswith('GPS'):
            gps_tag_ids[tag_name] = tag_id
    
    # 提取纬度
    latitude_ref = None
    latitude_val = None
    
    # 查找GPSLatitudeRef和GPSLatitude
    for key, value in gps_info_dict.items():
        if isinstance(key, int):
            # 如果是数值键，需要转换为标签名
            tag_name = ExifTags.GPSTAGS.get(key, key)
        else:
            tag_name = key
        
        if tag_name == 'GPSLatitudeRef':
            latitude_ref = value
        elif tag_name == 'GPSLatitude':
            latitude_val = value
    
    if latitude_val and latitude_ref:
        latitude = dms_to_decimal(latitude_val, latitude_ref)
        gps_info['latitude'] = latitude
    
    # 提取经度
    longitude_ref = None
    longitude_val = None
    
    for key, value in gps_info_dict.items():
        if isinstance(key, int):
            tag_name = ExifTags.GPSTAGS.get(key, key)
        else:
            tag_name = key
        
        if tag_name == 'GPSLongitudeRef':
            longitude_ref = value
        elif tag_name == 'GPSLongitude':
            longitude_val = value
    
    if longitude_val and longitude_ref:
        longitude = dms_to_decimal(longitude_val, longitude_ref)
        gps_info['longitude'] = longitude
    
    # 提取海拔
    altitude_val = None
    altitude_ref = None
    
    for key, value in gps_info_dict.items():
        if isinstance(key, int):
            tag_name = ExifTags.GPSTAGS.get(key, key)
        else:
            tag_name = key
        
        if tag_name == 'GPSAltitude':
            altitude_val = value
        elif tag_name == 'GPSAltitudeRef':
            altitude_ref = value
    
    if altitude_val:
        try:
            altitude = float(altitude_val)
            if altitude_ref == 1:  # 海平面以下
                altitude = -altitude
            gps_info['altitude'] = altitude
        except (ValueError, TypeError):
            pass
    
    # 必须有经纬度才返回有效数据
    if 'latitude' in gps_info and 'longitude' in gps_info:
        return gps_info
    
    return None


def extract_gps_from_image(image_path: str) -> Optional[Dict[str, float]]:
    """
    从图像文件中提取GPS坐标
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        包含经纬度的字典，格式为 {'latitude': float, 'longitude': float, 'altitude': float}
        如果没有GPS数据则返回None
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                gps_data = extract_gps_from_exif(exif_data)
                if gps_data:
                    print(f"[EXIF] 成功提取GPS数据 {os.path.basename(image_path)}: "
                          f"经度={gps_data.get('longitude', 0):.6f}, "
                          f"纬度={gps_data.get('latitude', 0):.6f}")
                else:
                    print(f"[EXIF] 图像 {os.path.basename(image_path)} 无GPS数据")
                return gps_data
            else:
                print(f"[EXIF] 图像 {os.path.basename(image_path)} 无EXIF数据")
    except Exception as e:
        print(f"[EXIF] 提取图像GPS数据失败 {image_path}: {e}")
    
    return None


def extract_gps_from_images(image_directory: str) -> List[Dict[str, float]]:
    """
    从目录中的所有图像文件中提取GPS坐标
    
    Args:
        image_directory: 图像目录路径
        
    Returns:
        包含所有有效GPS坐标的列表
    """
    gps_list = []
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    for filename in os.listdir(image_directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(image_directory, filename)
            gps_data = extract_gps_from_image(image_path)
            if gps_data:
                gps_data['source_file'] = filename
                gps_list.append(gps_data)
    
    return gps_list


def calculate_average_gps(gps_list: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    计算GPS坐标的平均值
    
    Args:
        gps_list: GPS坐标列表
        
    Returns:
        平均GPS坐标，格式为 {'latitude': float, 'longitude': float, 'altitude': float}
    """
    if not gps_list:
        return None
    
    latitudes = [gps['latitude'] for gps in gps_list if 'latitude' in gps]
    longitudes = [gps['longitude'] for gps in gps_list if 'longitude' in gps]
    altitudes = [gps['altitude'] for gps in gps_list if 'altitude' in gps]
    
    avg_gps = {}
    
    if latitudes:
        avg_gps['latitude'] = sum(latitudes) / len(latitudes)
    
    if longitudes:
        avg_gps['longitude'] = sum(longitudes) / len(longitudes)
    
    if altitudes:
        avg_gps['altitude'] = sum(altitudes) / len(altitudes)
    
    return avg_gps


def get_model_center_from_images(image_directory: str) -> Optional[Dict[str, float]]:
    """
    从图像目录中获取模型的中心GPS坐标
    
    Args:
        image_directory: 包含训练图像的目录
        
    Returns:
        模型中心GPS坐标，格式为 {'latitude': float, 'longitude': float, 'altitude': float}
    """
    if not os.path.exists(image_directory):
        return None
    
    # 提取所有图像的GPS数据
    gps_list = extract_gps_from_images(image_directory)
    
    if not gps_list:
        return None
    
    # 计算平均GPS坐标
    avg_gps = calculate_average_gps(gps_list)
    
    if avg_gps:
        print(f"从 {len(gps_list)} 张图像中提取GPS数据")
        print(f"模型中心坐标: 经度={avg_gps.get('longitude', 0):.6f}, "
              f"纬度={avg_gps.get('latitude', 0):.6f}, "
              f"海拔={avg_gps.get('altitude', 0):.2f}m")
    
    return avg_gps


def save_gps_info_to_json(gps_data: Dict[str, float], output_path: str):
    """
    将GPS数据保存为JSON文件
    
    Args:
        gps_data: GPS数据字典
        output_path: 输出JSON文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gps_data, f, indent=2, ensure_ascii=False)


def load_gps_info_from_json(json_path: str) -> Optional[Dict[str, float]]:
    """
    从JSON文件加载GPS数据
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        GPS数据字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载GPS JSON文件失败 {json_path}: {e}")
        return None


# 测试函数
if __name__ == "__main__":
    # 测试图像路径
    test_image_dir = "test_images"
    
    if os.path.exists(test_image_dir):
        gps_data = get_model_center_from_images(test_image_dir)
        if gps_data:
            print("GPS数据提取成功:")
            print(json.dumps(gps_data, indent=2))
        else:
            print("未找到GPS数据")
    else:
        print(f"测试目录 {test_image_dir} 不存在")