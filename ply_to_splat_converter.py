#!/usr/bin/env python3
"""
PLY到.splat格式转换器
将3D Gaussian Splatting的PLY格式转换为标准.splat格式（用于Cesium/GaussianSplats3D）
标准.splat格式：每个splat 32字节
- Position: 3 floats (12 bytes)
- Scale: 3 floats (12 bytes)
- Color: 4 uint8 (4 bytes) (R, G, B, A)
- Rotation: 4 uint8 (4 bytes) (Z, W, X, Y)
"""

import os
import struct
import numpy as np
from typing import Optional
import torch


def convert_gaussians_to_splat(gaussians, output_path: str, mask: Optional[np.ndarray] = None):
    """
    将GaussianModel转换为标准.splat格式
    
    Args:
        gaussians: GaussianModel对象
        output_path: 输出.splat文件路径
        mask: 可选掩码，用于转换特定分割的点
    """
    # 获取数据
    positions = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    opacities = gaussians.get_opacity.detach().cpu().numpy()  # [N, 1]
    scales = gaussians.get_scaling.detach().cpu().numpy()  # [N, 3]
    rotations = gaussians.get_rotation.detach().cpu().numpy()  # [N, 4] (quaternion w, x, y, z)
    features_dc = gaussians.get_features.detach().cpu().numpy()  # [N, 1, 3]
    
    # 应用掩码（如果提供）
    if mask is not None:
        if len(mask) != len(positions):
            raise ValueError(f"掩码长度({len(mask)})与点数({len(positions)})不匹配")
        
        positions = positions[mask]
        opacities = opacities[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        features_dc = features_dc[mask]
    
    num_gaussians = positions.shape[0]
    
    print(f"转换 {num_gaussians} 个高斯点到标准.splat格式...")
    if mask is not None:
        print(f"应用掩码过滤，原始点数: {len(mask)}, 过滤后点数: {num_gaussians}")
    
    # 1. 计算颜色 (SH DC -> RGB)
    # RGB = 0.5 + 0.28209479177387814 * DC
    SH_C0 = 0.28209479177387814
    colors = 0.5 + SH_C0 * features_dc[:, 0, :]  # [N, 3]
    colors = np.clip(colors, 0, 1) * 255
    colors = colors.astype(np.uint8)  # [N, 3]
    
    # 2. 计算不透明度 (0-1 -> 0-255)
    # opacities已经是sigmoid后的值(0-1)
    opacities_uint8 = np.clip(opacities, 0, 1) * 255
    opacities_uint8 = opacities_uint8.astype(np.uint8)
    if opacities_uint8.ndim == 2 and opacities_uint8.shape[1] == 1:
        opacities_uint8 = opacities_uint8.flatten()
        
    # 3. 计算旋转 (Quaternion -> uint8)
    # 归一化四元数
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (norms + 1e-6)
    
    # 映射到 [0, 255]
    rot_uint8 = np.clip(rotations * 128 + 128, 0, 255).astype(np.uint8)  # [N, 4] (w, x, y, z)
    
    # 准备二进制缓冲区
    # 格式：pos(12) + scale(12) + color(4) + rot(4) = 32 bytes
    
    # 使用numpy构建结构化数组以提高性能
    dtype = np.dtype([
        ('pos', 'f4', 3),
        ('scale', 'f4', 3),
        ('color', 'u1', 4),
        ('rot', 'u1', 4)
    ])
    
    data = np.zeros(num_gaussians, dtype=dtype)
    
    data['pos'] = positions.astype(np.float32)
    data['scale'] = scales.astype(np.float32)
    
    # Color: R, G, B, A
    data['color'][:, 0] = colors[:, 0]
    data['color'][:, 1] = colors[:, 1]
    data['color'][:, 2] = colors[:, 2]
    data['color'][:, 3] = opacities_uint8
    
    # Rotation: GaussianSplats3D viewer expects:
    # Byte 28: Z (index 3 in w,x,y,z if w is 0? Wait)
    # Viewer mapping:
    #   x = (uBuffer[30] - 128) / 128
    #   y = (uBuffer[31] - 128) / 128
    #   z = (uBuffer[28] - 128) / 128
    #   w = (uBuffer[29] - 128) / 128
    #
    # Input rotation is (w, x, y, z)
    # We want viewer's (x, y, z, w) to match input's (x, y, z, w)
    # So:
    #   Viewer's x (from byte 30) = Input x
    #   Viewer's y (from byte 31) = Input y
    #   Viewer's z (from byte 28) = Input z
    #   Viewer's w (from byte 29) = Input w
    
    data['rot'][:, 0] = rot_uint8[:, 3]  # Byte 28 -> Input z
    data['rot'][:, 1] = rot_uint8[:, 0]  # Byte 29 -> Input w
    data['rot'][:, 2] = rot_uint8[:, 1]  # Byte 30 -> Input x
    data['rot'][:, 3] = rot_uint8[:, 2]  # Byte 31 -> Input y
    
    # 写入文件
    with open(output_path, 'wb') as f:
        data.tofile(f)
    
    print(f".splat文件已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


def convert_ply_to_splat(ply_path: str, output_path: str):
    """
    从PLY文件转换为标准.splat格式
    
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
        
        # 旋转四元数 (w, x, y, z)
        rotations = np.stack([
            np.array(vertex['rot_0']),
            np.array(vertex['rot_1']),
            np.array(vertex['rot_2']),
            np.array(vertex['rot_3'])
        ], axis=1)
        
        # 颜色 (从DC项计算)
        # f_dc_0, f_dc_1, f_dc_2
        dc_0 = np.array(vertex['f_dc_0'])
        dc_1 = np.array(vertex['f_dc_1'])
        dc_2 = np.array(vertex['f_dc_2'])
        
        SH_C0 = 0.28209479177387814
        colors_r = 0.5 + SH_C0 * dc_0
        colors_g = 0.5 + SH_C0 * dc_1
        colors_b = 0.5 + SH_C0 * dc_2
        
        colors = np.stack([colors_r, colors_g, colors_b], axis=1)
        colors = np.clip(colors, 0, 1) * 255
        colors = colors.astype(np.uint8)
        
        # 不透明度
        opacities = np.array(vertex['opacity'])
        # 假设ply中的opacity已经过了sigmoid (0-1)，或者需要sigmoid?
        # 通常ply存储的是logit，需要sigmoid。但这里为了兼容性，先检查范围
        # 如果范围在0-1之间，假设已处理。如果很大/很小，可能是logit
        if opacities.max() > 1.0 or opacities.min() < 0.0:
            opacities = 1.0 / (1.0 + np.exp(-opacities))
            
        opacities_uint8 = np.clip(opacities, 0, 1) * 255
        opacities_uint8 = opacities_uint8.astype(np.uint8)
        
        num_gaussians = positions.shape[0]
        print(f"从PLY读取 {num_gaussians} 个高斯点...")
        
        # 3. 计算旋转 (Quaternion -> uint8)
        norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / (norms + 1e-6)
        rot_uint8 = np.clip(rotations * 128 + 128, 0, 255).astype(np.uint8)
        
        # 准备数据
        dtype = np.dtype([
            ('pos', 'f4', 3),
            ('scale', 'f4', 3),
            ('color', 'u1', 4),
            ('rot', 'u1', 4)
        ])
        
        data = np.zeros(num_gaussians, dtype=dtype)
        
        data['pos'] = positions.astype(np.float32)
        data['scale'] = scales.astype(np.float32)
        
        # Color: R, G, B, A
        data['color'][:, 0] = colors[:, 0]
        data['color'][:, 1] = colors[:, 1]
        data['color'][:, 2] = colors[:, 2]
        data['color'][:, 3] = opacities_uint8
        
        # Rotation: Z, W, X, Y
        # Input (w, x, y, z)
        data['rot'][:, 0] = rot_uint8[:, 3]  # Z
        data['rot'][:, 1] = rot_uint8[:, 0]  # W
        data['rot'][:, 2] = rot_uint8[:, 1]  # X
        data['rot'][:, 3] = rot_uint8[:, 2]  # Y
        
        # 写入文件
        with open(output_path, 'wb') as f:
            data.tofile(f)
        
        print(f".splat文件已保存到: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"PLY转换失败: {e}")
        raise


if __name__ == '__main__':
    # 测试代码
    print("PLY到.splat转换器模块加载成功")

