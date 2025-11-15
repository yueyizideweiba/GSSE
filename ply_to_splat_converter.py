#!/usr/bin/env python3
"""
PLY到.splat格式转换器
将3D Gaussian Splatting的PLY格式转换为.splat格式（用于Cesium）
"""

import os
import struct
import numpy as np
from typing import Optional
import torch


def convert_gaussians_to_splat(gaussians, output_path: str, mask: Optional[np.ndarray] = None):
    """
    将GaussianModel转换为.splat格式
    
    Args:
        gaussians: GaussianModel对象
        output_path: 输出.splat文件路径
        mask: 可选掩码，用于转换特定分割的点
    """
    # 获取数据
    positions = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    opacities = gaussians.get_opacity.detach().cpu().numpy()  # [N, 1]
    scales = gaussians.get_scaling.detach().cpu().numpy()  # [N, 3]
    rotations = gaussians.get_rotation.detach().cpu().numpy()  # [N, 4] (quaternion)
    features_dc = gaussians.get_features.detach().cpu().numpy()  # [N, 1, 3]
    features_rest = gaussians.get_features[:, 1:, :].detach().cpu().numpy()  # [N, 15, 3]
    
    # 应用掩码（如果提供）
    if mask is not None:
        if len(mask) != len(positions):
            raise ValueError(f"掩码长度({len(mask)})与点数({len(positions)})不匹配")
        
        positions = positions[mask]
        opacities = opacities[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        features_dc = features_dc[mask]
        features_rest = features_rest[mask]
    
    num_gaussians = positions.shape[0]
    
    print(f"转换 {num_gaussians} 个高斯点到.splat格式...")
    if mask is not None:
        print(f"应用掩码过滤，原始点数: {len(mask)}, 过滤后点数: {num_gaussians}")
    
    # 打开输出文件
    with open(output_path, 'wb') as f:
        # 写入每个高斯点
        for i in range(num_gaussians):
            # 位置 (3 floats) - 确保转换为Python标量
            pos = positions[i]
            f.write(struct.pack('fff', 
                float(pos[0]),
                float(pos[1]),
                float(pos[2])
            ))
            
            # 缩放 (3 floats) - 确保转换为Python标量
            scale = scales[i]
            f.write(struct.pack('fff',
                float(scale[0]),
                float(scale[1]),
                float(scale[2])
            ))
            
            # 旋转四元数 (4 floats) - 确保转换为Python标量
            rot = rotations[i]
            f.write(struct.pack('ffff',
                float(rot[0]),
                float(rot[1]),
                float(rot[2]),
                float(rot[3])
            ))
            
            # 球谐系数 (48 floats for RGB)
            # DC项 (1个) - 确保转换为Python标量
            sh_dc = features_dc[i, 0, :]  # [3]
            f.write(struct.pack('fff',
                float(sh_dc[0]),
                float(sh_dc[1]),
                float(sh_dc[2])
            ))
            
            # Rest项 (15个) - 确保转换为Python标量
            sh_rest = features_rest[i].flatten()  # [45]
            for j in range(45):
                f.write(struct.pack('f', float(sh_rest[j])))
            
            # 不透明度 (1 float) - 确保转换为Python标量
            opacity = opacities[i]
            if opacity.ndim > 0:
                opacity_val = float(opacity[0])
            else:
                opacity_val = float(opacity)
            f.write(struct.pack('f', opacity_val))
    
    print(f".splat文件已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def convert_ply_to_splat(ply_path: str, output_path: str):
    """
    从PLY文件转换为.splat格式
    
    Args:
        ply_path: 输入PLY文件路径
        output_path: 输出.splat文件路径
    """
    try:
        from plyfile import PlyData
        
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        
        # 提取数据
        vertex = plydata['vertex']
        
        positions = np.stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ], axis=1)
        
        # 缩放
        scales = np.stack([
            np.array(vertex['scale_0']),
            np.array(vertex['scale_1']),
            np.array(vertex['scale_2'])
        ], axis=1)
        
        # 旋转四元数
        rotations = np.stack([
            np.array(vertex['rot_0']),
            np.array(vertex['rot_1']),
            np.array(vertex['rot_2']),
            np.array(vertex['rot_3'])
        ], axis=1)
        
        # 球谐系数
        # PLY格式中：f_dc_0, f_dc_1, f_dc_2 是DC项（3个）
        # f_rest_0 到 f_rest_44 是Rest项（45个），共48个系数
        sh_features = []
        for i in range(48):
            if i < 3:
                # DC项
                sh_features.append(np.array(vertex[f'f_dc_{i}']))
            else:
                # Rest项
                sh_features.append(np.array(vertex[f'f_rest_{i-3}']))
        sh_features = np.stack(sh_features, axis=1)
        
        # 不透明度
        opacities = np.array(vertex['opacity'])
        
        num_gaussians = positions.shape[0]
        print(f"从PLY读取 {num_gaussians} 个高斯点...")
        
        # 写入.splat文件
        with open(output_path, 'wb') as f:
            for i in range(num_gaussians):
                # 位置
                f.write(struct.pack('fff',
                    float(positions[i, 0]),
                    float(positions[i, 1]),
                    float(positions[i, 2])
                ))
                
                # 缩放
                f.write(struct.pack('fff',
                    float(scales[i, 0]),
                    float(scales[i, 1]),
                    float(scales[i, 2])
                ))
                
                # 旋转
                f.write(struct.pack('ffff',
                    float(rotations[i, 0]),
                    float(rotations[i, 1]),
                    float(rotations[i, 2]),
                    float(rotations[i, 3])
                ))
                
                # 球谐系数
                for j in range(48):
                    f.write(struct.pack('f', float(sh_features[i, j])))
                
                # 不透明度
                f.write(struct.pack('f', float(opacities[i])))
        
        print(f".splat文件已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"PLY转换失败: {e}")
        raise


if __name__ == '__main__':
    # 测试代码
    print("PLY到.splat转换器模块加载成功")

