#!/usr/bin/env python3
"""
3D Gaussian Splatting 编辑模块

"""

import torch
import numpy as np
from typing import List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass
from collections import deque
import copy


class EditType(Enum):
    """编辑类型"""
    NONE = 0
    BOX = 1
    RECT = 3
    BRUSH = 4
    LASSO = 6
    PAINT = 7  # 颜色绘制模式


class EditSelectOpType(Enum):
    """选择操作类型"""
    SET = 0      # 设置为选中
    REMOVE = 1   # 移除选中
    ADD = 2      # 添加到选中
    DEFAULT = 3


# 状态标志位
NORMAL_STATE = 0x00
SELECT_STATE = 0x01
DELETE_STATE = 0x02
HIDE_STATE = 0x04
LOCK_STATE = 0x08


@dataclass
class BoundingBox:
    """轴对齐包围盒"""
    min_point: np.ndarray  # [3]
    max_point: np.ndarray  # [3]
    
    def contains(self, point: np.ndarray) -> bool:
        """检查点是否在包围盒内"""
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
    
    @classmethod
    def from_points(cls, points: np.ndarray) -> 'BoundingBox':
        """从点集创建包围盒"""
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        return cls(min_point=min_point, max_point=max_point)


class EditOperation:
    """编辑操作基类"""
    def apply(self):
        """应用操作"""
        raise NotImplementedError
    
    def undo(self):
        """撤销操作"""
        raise NotImplementedError


class SplatSelectionOp(EditOperation):
    """Splat选择操作（优化版本）"""
    def __init__(self, gaussians, op_type: EditSelectOpType, filter_func: Optional[Callable[[int], bool]] = None, 
                 precomputed_mask: Optional[torch.Tensor] = None):
        """
        Args:
            gaussians: GaussianModel实例
            op_type: 选择操作类型
            filter_func: 过滤函数（已弃用，优先使用precomputed_mask）
            precomputed_mask: 预计算的mask tensor（GPU，bool类型），形状为[num_points]
        """
        self.gaussians = gaussians
        self.op_type = op_type
        self.filter_func = filter_func
        self.precomputed_mask = precomputed_mask
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """应用选择操作（GPU加速版本）"""
        if not hasattr(self.gaussians, '_state_flags'):
            # 初始化状态标志
            num_points = self.gaussians.get_xyz.shape[0]
            self.gaussians._state_flags = torch.zeros(num_points, dtype=torch.uint8, device=self.gaussians.get_xyz.device)
        
        flags = self.gaussians._state_flags
        device = flags.device
        
        # 如果提供了预计算的mask，直接使用（最快）
        if self.precomputed_mask is not None:
            affected_mask = self.precomputed_mask.to(device)
        elif self.filter_func is not None:
            # 否则使用filter_func（较慢）
            all_indices = torch.arange(len(flags), device=device)
            valid_mask = ((flags & DELETE_STATE) == 0) & ((flags & HIDE_STATE) == 0)
            valid_indices = all_indices[valid_mask]
            
            if len(valid_indices) == 0:
                return
            
            # 批量检查filter_func
            batch_size = 10000
            affected_list = []
            for i in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[i:i+batch_size]
                batch_np = batch_indices.cpu().numpy()
                batch_mask = np.array([self.filter_func(int(idx)) for idx in batch_np])
                affected_batch = batch_indices[batch_mask]
                affected_list.append(affected_batch)
            
            if len(affected_list) > 0:
                affected_tensor = torch.cat(affected_list)
                affected_mask = torch.zeros(len(flags), dtype=torch.bool, device=device)
                affected_mask[affected_tensor] = True
            else:
                affected_mask = torch.zeros(len(flags), dtype=torch.bool, device=device)
        else:
            return
        
        # 过滤掉已删除和隐藏的点
        valid_mask = ((flags & DELETE_STATE) == 0) & ((flags & HIDE_STATE) == 0)
        affected_mask = affected_mask & valid_mask
        
        # 找出受影响的索引
        affected_tensor = torch.where(affected_mask)[0]
        if len(affected_tensor) == 0:
            return
        
        self.affected_indices = affected_tensor.cpu().tolist()
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
        
        # 批量更新状态标志（GPU向量化操作）
        with torch.no_grad():
            if self.op_type == EditSelectOpType.ADD:
                flags.data[affected_tensor] |= SELECT_STATE
            elif self.op_type == EditSelectOpType.REMOVE:
                flags.data[affected_tensor] &= ~SELECT_STATE
            elif self.op_type == EditSelectOpType.SET:
                flags.data[affected_tensor] |= SELECT_STATE
            else:  # SET for unselected
                flags.data[affected_tensor] &= ~SELECT_STATE
    
    def undo(self):
        """撤销选择操作"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class DeleteSplatOp(EditOperation):
    """删除Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.deleted_indices = []
        self.backup_data = {}
    
    def apply(self):
        """删除选中的splats（排除锁定的点）"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        self.deleted_indices = []
        
        # 找到所有选中且未锁定的点
        selected_mask = (flags & SELECT_STATE) != 0
        unlocked_mask = (flags & LOCK_STATE) == 0
        deletable_mask = selected_mask & unlocked_mask
        device = flags.device
        deleted_tensor = torch.where(deletable_mask)[0]
        self.deleted_indices = deleted_tensor.cpu().tolist()
        
        if not self.deleted_indices:
            return
        
        # 确保索引在正确的设备上
        deleted_indices_tensor = deleted_tensor.to(device)
        
        # 备份数据
        with torch.no_grad():
            self.backup_data = {
                'xyz': self.gaussians._xyz[deleted_indices_tensor].clone(),
                'features_dc': self.gaussians._features_dc[deleted_indices_tensor].clone(),
                'features_rest': self.gaussians._features_rest[deleted_indices_tensor].clone(),
                'scaling': self.gaussians._scaling[deleted_indices_tensor].clone(),
                'rotation': self.gaussians._rotation[deleted_indices_tensor].clone(),
                'opacity': self.gaussians._opacity[deleted_indices_tensor].clone(),
            }
        
        # 标记为删除状态，并清除选择状态
        with torch.no_grad():
            flags.data[deleted_indices_tensor] |= DELETE_STATE
            flags.data[deleted_indices_tensor] &= ~SELECT_STATE  # 清除选择状态
    
    def undo(self):
        """撤销删除"""
        if not self.deleted_indices:
            return
        
        flags = self.gaussians._state_flags
        device = flags.device
        deleted_indices_tensor = torch.tensor(self.deleted_indices, dtype=torch.long, device=device)
        with torch.no_grad():
            flags.data[deleted_indices_tensor] &= ~DELETE_STATE


class HideSplatOp(EditOperation):
    """隐藏Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """隐藏选中的splats"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        selected_mask = (flags & SELECT_STATE) != 0
        affected_tensor = torch.where(selected_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
            # 设置隐藏状态
            flags.data[affected_tensor] |= HIDE_STATE
    
    def undo(self):
        """撤销隐藏"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class UnhideSplatOp(EditOperation):
    """显示Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """显示选中的splats"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        selected_mask = (flags & SELECT_STATE) != 0
        affected_tensor = torch.where(selected_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
            # 清除隐藏状态
            flags.data[affected_tensor] &= ~HIDE_STATE
    
    def undo(self):
        """撤销显示"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class LockSplatOp(EditOperation):
    """锁定Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """锁定选中的splats"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        selected_mask = (flags & SELECT_STATE) != 0
        affected_tensor = torch.where(selected_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
            # 设置锁定状态
            flags.data[affected_tensor] |= LOCK_STATE
    
    def undo(self):
        """撤销锁定"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class UnlockSplatOp(EditOperation):
    """解锁Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """解锁选中的splats"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        selected_mask = (flags & SELECT_STATE) != 0
        affected_tensor = torch.where(selected_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
            # 清除锁定状态
            flags.data[affected_tensor] &= ~LOCK_STATE
    
    def undo(self):
        """撤销解锁"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class UnlockAllSplatOp(EditOperation):
    """解锁所有Splat操作"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.affected_indices = []
        self.old_states = {}
    
    def apply(self):
        """解锁所有锁定的splats"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        # 找到所有锁定的点
        locked_mask = (flags & LOCK_STATE) != 0
        affected_tensor = torch.where(locked_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        # 备份旧状态
        with torch.no_grad():
            self.old_states = {int(idx): flags[idx].item() for idx in affected_tensor}
            # 清除所有锁定状态
            flags.data[affected_tensor] &= ~LOCK_STATE
    
    def undo(self):
        """撤销解锁所有"""
        flags = self.gaussians._state_flags
        device = flags.device
        with torch.no_grad():
            for idx, old_state in self.old_states.items():
                idx_tensor = torch.tensor([idx], dtype=torch.long, device=device)
                flags.data[idx_tensor] = old_state


class ColorAdjustmentOp(EditOperation):
    """颜色调整操作"""
    def __init__(self, gaussians, rgb_adjustment, brightness, transparency):
        self.gaussians = gaussians
        self.rgb_adjustment = rgb_adjustment  # [r, g, b] delta
        self.brightness = brightness
        self.transparency = transparency
        self.affected_indices = []
        self.backup_features_dc = None
        self.backup_opacity = None
    
    def apply(self):
        """应用颜色调整（排除锁定的点）"""
        selected_mask = (self.gaussians._state_flags & SELECT_STATE) != 0
        unlocked_mask = (self.gaussians._state_flags & LOCK_STATE) == 0
        adjustable_mask = selected_mask & unlocked_mask
        affected_tensor = torch.where(adjustable_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        device = self.gaussians.get_xyz.device
        
        # 备份数据
        with torch.no_grad():
            self.backup_features_dc = self.gaussians._features_dc[affected_tensor].clone()
            self.backup_opacity = self.gaussians._opacity[affected_tensor].clone()
            
            # 调整颜色
            rgb_delta = torch.tensor(self.rgb_adjustment, dtype=torch.float32, device=device).reshape(1, 1, 3)
            self.gaussians._features_dc.data[affected_tensor] += rgb_delta
            
            # 调整亮度
            if self.brightness != 0:
                brightness_delta = torch.tensor([self.brightness], dtype=torch.float32, device=device).reshape(1, 1, 1)
                self.gaussians._features_dc.data[affected_tensor] += brightness_delta
            
            # 调整透明度
            if self.transparency != 0:
                transparency_delta = torch.tensor([self.transparency], dtype=torch.float32, device=device).reshape(1, 1)
                self.gaussians._opacity.data[affected_tensor] += transparency_delta
                # 限制范围在合理区间
                self.gaussians._opacity.data[affected_tensor] = torch.clamp(
                    self.gaussians._opacity.data[affected_tensor], -10, 10
                )
    
    def undo(self):
        """撤销颜色调整"""
        if self.backup_features_dc is None:
            return
        
        device = self.gaussians.get_xyz.device
        affected_tensor = torch.tensor(self.affected_indices, dtype=torch.long, device=device)
        
        with torch.no_grad():
            self.gaussians._features_dc.data[affected_tensor] = self.backup_features_dc
            self.gaussians._opacity.data[affected_tensor] = self.backup_opacity


class PaintColorOp(EditOperation):
    """颜色绘制操作（混合颜色）"""
    def __init__(self, gaussians, color, mix_weight=0.5):
        self.gaussians = gaussians
        self.color = color  # [r, g, b]
        self.mix_weight = mix_weight  # 混合权重，0-1
        self.affected_indices = []
        self.backup_features_dc = None
    
    def apply(self):
        """应用颜色绘制（排除锁定的点）"""
        selected_mask = (self.gaussians._state_flags & SELECT_STATE) != 0
        unlocked_mask = (self.gaussians._state_flags & LOCK_STATE) == 0
        paintable_mask = selected_mask & unlocked_mask
        affected_tensor = torch.where(paintable_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        device = self.gaussians.get_xyz.device
        
        # 备份数据
        with torch.no_grad():
            self.backup_features_dc = self.gaussians._features_dc[affected_tensor].clone()
            
            # 混合颜色
            target_color = torch.tensor(self.color, dtype=torch.float32, device=device).reshape(1, 1, 3)
            current_color = self.gaussians._features_dc[affected_tensor]
            mixed_color = current_color * (1 - self.mix_weight) + target_color * self.mix_weight
            self.gaussians._features_dc.data[affected_tensor] = mixed_color
    
    def undo(self):
        """撤销颜色绘制"""
        if self.backup_features_dc is None:
            return
        
        device = self.gaussians.get_xyz.device
        affected_tensor = torch.tensor(self.affected_indices, dtype=torch.long, device=device)
        
        with torch.no_grad():
            self.gaussians._features_dc.data[affected_tensor] = self.backup_features_dc


class TransformOp(EditOperation):
    """变换操作（包括平移、旋转、缩放）"""
    def __init__(self, gaussians, translation=None, rotation=None, scale=None, center=None):
        self.gaussians = gaussians
        self.translation = translation  # [x, y, z]
        self.rotation = rotation  # 四元数 [w, x, y, z]
        self.scale = scale  # [sx, sy, sz]
        self.center = center  # 旋转/缩放中心点
        self.affected_indices = []
        self.backup_xyz = None
        self.backup_scaling = None
        self.backup_rotation = None
    
    def apply(self):
        """应用变换（排除锁定的点）"""
        selected_mask = (self.gaussians._state_flags & SELECT_STATE) != 0
        unlocked_mask = (self.gaussians._state_flags & LOCK_STATE) == 0
        transformable_mask = selected_mask & unlocked_mask
        affected_tensor = torch.where(transformable_mask)[0]
        self.affected_indices = affected_tensor.cpu().tolist()
        
        if not self.affected_indices:
            return
        
        device = self.gaussians.get_xyz.device
        
        # 备份数据
        with torch.no_grad():
            self.backup_xyz = self.gaussians._xyz[affected_tensor].clone()
            self.backup_scaling = self.gaussians._scaling[affected_tensor].clone()
            self.backup_rotation = self.gaussians._rotation[affected_tensor].clone()
            
            # 应用平移
            if self.translation is not None:
                translation_tensor = torch.tensor(self.translation, dtype=torch.float32, device=device).reshape(1, 3)
                self.gaussians._xyz.data[affected_tensor] += translation_tensor
            
            # 应用缩放
            if self.scale is not None:
                scale_tensor = torch.tensor(self.scale, dtype=torch.float32, device=device).reshape(1, 3)
                scale_log = torch.log(torch.clamp(scale_tensor, min=1e-6))
                self.gaussians._scaling.data[affected_tensor] += scale_log
            
            # 应用旋转（四元数乘法）
            if self.rotation is not None:
                # 简化实现：直接设置旋转
                rotation_tensor = torch.tensor(self.rotation, dtype=torch.float32, device=device).reshape(1, 4)
                rotation_tensor = rotation_tensor / (torch.norm(rotation_tensor) + 1e-8)
                self.gaussians._rotation.data[affected_tensor] = rotation_tensor.expand(len(affected_tensor), 4)
    
    def undo(self):
        """撤销变换"""
        if self.backup_xyz is None:
            return
        
        device = self.gaussians.get_xyz.device
        affected_tensor = torch.tensor(self.affected_indices, dtype=torch.long, device=device)
        
        with torch.no_grad():
            self.gaussians._xyz.data[affected_tensor] = self.backup_xyz
            self.gaussians._scaling.data[affected_tensor] = self.backup_scaling
            self.gaussians._rotation.data[affected_tensor] = self.backup_rotation


class DuplicateSplatOp(EditOperation):
    """复制Splat操作"""
    def __init__(self, gaussians, selected_indices):
        self.gaussians = gaussians
        # 保存为CPU上的列表，避免设备问题
        if isinstance(selected_indices, torch.Tensor):
            self.selected_indices = selected_indices.cpu().tolist()
        else:
            self.selected_indices = selected_indices if isinstance(selected_indices, list) else list(selected_indices)
        self.num_original_points = gaussians.get_xyz.shape[0]
        self.num_duplicated = len(self.selected_indices)
        self.original_num_points = None
    
    def apply(self):
        """复制选中的splats"""
        # 确保索引在正确的设备上
        device = self.gaussians.get_xyz.device
        # 从CPU列表转换为GPU tensor
        selected_indices = torch.tensor(self.selected_indices, dtype=torch.long, device=device)
        
        # 记录原始点数
        self.original_num_points = self.gaussians.get_xyz.shape[0]
        
        # 复制数据
        with torch.no_grad():
            new_xyz = self.gaussians._xyz[selected_indices].clone()
            new_features_dc = self.gaussians._features_dc[selected_indices].clone()
            new_features_rest = self.gaussians._features_rest[selected_indices].clone()
            new_scaling = self.gaussians._scaling[selected_indices].clone()
            new_rotation = self.gaussians._rotation[selected_indices].clone()
            new_opacity = self.gaussians._opacity[selected_indices].clone()
            
            # 添加到模型
            self.gaussians._xyz = torch.cat([self.gaussians._xyz, new_xyz], dim=0)
            self.gaussians._features_dc = torch.cat([self.gaussians._features_dc, new_features_dc], dim=0)
            self.gaussians._features_rest = torch.cat([self.gaussians._features_rest, new_features_rest], dim=0)
            self.gaussians._scaling = torch.cat([self.gaussians._scaling, new_scaling], dim=0)
            self.gaussians._rotation = torch.cat([self.gaussians._rotation, new_rotation], dim=0)
            self.gaussians._opacity = torch.cat([self.gaussians._opacity, new_opacity], dim=0)
            
            # 更新状态标志
            num_new = len(selected_indices)
            if hasattr(self.gaussians, '_state_flags'):
                new_flags = torch.zeros(num_new, dtype=torch.uint8, device=device)
                self.gaussians._state_flags = torch.cat([self.gaussians._state_flags, new_flags], dim=0)
            
            # 更新max_radii2D
            if hasattr(self.gaussians, 'max_radii2D'):
                new_max_radii2D = self.gaussians.max_radii2D[selected_indices].clone()
                self.gaussians.max_radii2D = torch.cat([self.gaussians.max_radii2D, new_max_radii2D], dim=0)
    
    def undo(self):
        """撤销复制"""
        if self.original_num_points is None:
            return
        
        # 移除复制的点（保留原始点）
        with torch.no_grad():
            self.gaussians._xyz = self.gaussians._xyz[:self.original_num_points]
            self.gaussians._features_dc = self.gaussians._features_dc[:self.original_num_points]
            self.gaussians._features_rest = self.gaussians._features_rest[:self.original_num_points]
            self.gaussians._scaling = self.gaussians._scaling[:self.original_num_points]
            self.gaussians._rotation = self.gaussians._rotation[:self.original_num_points]
            self.gaussians._opacity = self.gaussians._opacity[:self.original_num_points]
            
            if hasattr(self.gaussians, '_state_flags'):
                self.gaussians._state_flags = self.gaussians._state_flags[:self.original_num_points]
            
            if hasattr(self.gaussians, 'max_radii2D'):
                self.gaussians.max_radii2D = self.gaussians.max_radii2D[:self.original_num_points]


class ResetSplatOp(EditOperation):
    """重置Splat操作（取消所有选择和删除标记）"""
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.old_flags = None
    
    def apply(self):
        """重置所有状态"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        with torch.no_grad():
            self.old_flags = flags.clone()
            flags.data.fill_(NORMAL_STATE)
    
    def undo(self):
        """撤销重置"""
        if self.old_flags is not None:
            with torch.no_grad():
                self.gaussians._state_flags.data.copy_(self.old_flags)


class UndoRedoSystem:
    """撤销/重做系统"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.undo_stack = deque(maxlen=100)
            cls._instance.redo_stack = deque(maxlen=100)
        return cls._instance
    
    def add(self, operation: EditOperation):
        """添加操作"""
        operation.apply()
        self.undo_stack.append(operation)
        self.redo_stack.clear()  # 清除重做栈
    
    def undo(self):
        """撤销"""
        if self.undo_stack:
            op = self.undo_stack.pop()
            op.undo()
            self.redo_stack.append(op)
            return True
        return False
    
    def redo(self):
        """重做"""
        if self.redo_stack:
            op = self.redo_stack.pop()
            op.apply()
            self.undo_stack.append(op)
            return True
        return False
    
    def can_undo(self) -> bool:
        """是否可以撤销"""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """是否可以重做"""
        return len(self.redo_stack) > 0
    
    def clear(self):
        """清空历史"""
        self.undo_stack.clear()
        self.redo_stack.clear()


@dataclass
class Transform:
    """简单的变换类"""
    position: np.ndarray = None  # [x, y, z]
    rotation: np.ndarray = None  # 四元数 [w, x, y, z]
    scale: np.ndarray = None     # [sx, sy, sz]
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0])
        if self.rotation is None:
            self.rotation = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
        if self.scale is None:
            self.scale = np.array([1.0, 1.0, 1.0])


class Pivot:
    """Pivot点（变换中心点）"""
    def __init__(self, transform: Transform = None):
        self.transform = transform if transform is not None else Transform()
    
    def set_position(self, position: np.ndarray):
        """设置pivot点位置"""
        self.transform.position = np.asarray(position)
    
    def get_position(self) -> np.ndarray:
        """获取pivot点位置"""
        return self.transform.position
    
    def set_transform(self, transform: Transform):
        """设置完整变换"""
        self.transform = transform
    
    def get_transform(self) -> Transform:
        """获取完整变换"""
        return self.transform


class GaussianEditor:
    """Gaussian编辑器"""
    
    def __init__(self, gaussians):
        """
        初始化编辑器
        
        Args:
            gaussians: GaussianModel实例
        """
        self.gaussians = gaussians
        self.edit_type = EditType.NONE
        self.bounding_box = None
        self.brush_points = []  # 笔刷路径点
        self.brush_thickness = 60
        self.pivot = Pivot()  # Pivot点
        
        # 初始化状态标志
        if gaussians is not None:
            num_points = gaussians.get_xyz.shape[0]
            if not hasattr(gaussians, '_state_flags'):
                gaussians._state_flags = torch.zeros(
                    num_points, dtype=torch.uint8, device=gaussians.get_xyz.device
                )
    
    def set_edit_type(self, edit_type: EditType):
        """设置编辑类型"""
        self.edit_type = edit_type
    
    def set_bounding_box(self, min_point: np.ndarray, max_point: np.ndarray):
        """设置包围盒"""
        self.bounding_box = BoundingBox(min_point=min_point, max_point=max_point)
    
    def set_brush_path(self, points: List[Tuple[int, int]]):
        """设置笔刷路径（屏幕坐标）"""
        self.brush_points = points
    
    def select_by_box(self, min_point: np.ndarray, max_point: np.ndarray, 
                     op_type: EditSelectOpType = EditSelectOpType.SET):
        """通过包围盒选择"""
        bbox = BoundingBox(min_point=min_point, max_point=max_point)
        
        def filter_func(i):
            point = self.gaussians.get_xyz[i].cpu().numpy()
            return bbox.contains(point)
        
        op = SplatSelectionOp(self.gaussians, op_type, filter_func)
        UndoRedoSystem().add(op)
    
    def select_by_brush(self, brush_points: List[Tuple[int, int]], 
                       camera_view, screen_width: int, screen_height: int,
                       op_type: EditSelectOpType = EditSelectOpType.SET):
        """
        通过笔刷选择（在2D屏幕上绘制路径，选择投影到该路径的3D点）
        
        Args:
            brush_points: 屏幕坐标点列表 [(x, y), ...]
            camera_view: 相机视图对象
            screen_width: 屏幕宽度
            screen_height: 屏幕高度
            op_type: 选择操作类型
        """
        # TODO: 实现笔刷选择逻辑
        # 需要将屏幕坐标投影到3D空间，然后选择在笔刷路径附近的点
        pass
    
    def select_all(self):
        """全选（GPU加速版本）"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        device = flags.device
        
        # 创建全选mask（排除已删除和隐藏的点）
        valid_mask = ((flags & DELETE_STATE) == 0) & ((flags & HIDE_STATE) == 0)
        
        op = SplatSelectionOp(self.gaussians, EditSelectOpType.SET, precomputed_mask=valid_mask)
        UndoRedoSystem().add(op)
    
    def select_inverse(self):
        """反选（GPU加速版本）"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        device = flags.device
        
        # 先记录当前选中的点和未选中的点
        valid_mask = ((flags & DELETE_STATE) == 0) & ((flags & HIDE_STATE) == 0)
        currently_selected = ((flags & SELECT_STATE) != 0) & valid_mask
        currently_unselected = ((flags & SELECT_STATE) == 0) & valid_mask
        
        # 先取消当前选中的点
        if torch.any(currently_selected):
            op1 = SplatSelectionOp(self.gaussians, EditSelectOpType.REMOVE, precomputed_mask=currently_selected)
            UndoRedoSystem().add(op1)
        
        # 然后选中之前未选中的点
        if torch.any(currently_unselected):
            op2 = SplatSelectionOp(self.gaussians, EditSelectOpType.SET, precomputed_mask=currently_unselected)
            UndoRedoSystem().add(op2)
    
    def select_none(self):
        """取消选择（GPU加速版本）"""
        if not hasattr(self.gaussians, '_state_flags'):
            return
        
        flags = self.gaussians._state_flags
        device = flags.device
        
        # 创建mask：所有当前选中的点
        selected_mask = (flags & SELECT_STATE) != 0
        
        op = SplatSelectionOp(self.gaussians, EditSelectOpType.REMOVE, precomputed_mask=selected_mask)
        UndoRedoSystem().add(op)
    
    def delete_selected(self):
        """删除选中的splats"""
        op = DeleteSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def reset_selection(self):
        """重置所有选择和删除标记"""
        op = ResetSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def get_selected_indices(self) -> torch.Tensor:
        """获取选中的索引"""
        if not hasattr(self.gaussians, '_state_flags'):
            device = self.gaussians.get_xyz.device
            return torch.tensor([], dtype=torch.long, device=device)
        
        flags = self.gaussians._state_flags
        selected_mask = (flags & SELECT_STATE) != 0
        indices = torch.where(selected_mask)[0]
        # 确保索引在正确的设备上
        if indices.device != self.gaussians.get_xyz.device:
            indices = indices.to(self.gaussians.get_xyz.device)
        return indices
    
    def get_visible_indices(self) -> torch.Tensor:
        """获取可见的索引（排除删除和隐藏的点）"""
        if not hasattr(self.gaussians, '_state_flags'):
            num_points = self.gaussians.get_xyz.shape[0]
            return torch.arange(num_points, device=self.gaussians.get_xyz.device)
        
        flags = self.gaussians._state_flags
        visible_mask = ((flags & DELETE_STATE) == 0) & ((flags & HIDE_STATE) == 0)
        return torch.where(visible_mask)[0]
    
    def apply_filter_to_render(self):
        """应用过滤到渲染（只渲染可见的点）"""
        visible_indices = self.get_visible_indices()
        
        # 创建过滤后的数据
        # 注意：这里不修改原始数据，而是在渲染时使用索引
        return visible_indices
    
    def export_selected(self, output_path: str):
        """导出选中的splats到PLY文件"""
        selected_indices = self.get_selected_indices()
        
        if len(selected_indices) == 0:
            raise ValueError("没有选中的点")
        
        # 直接写 PLY（子集）
        from plyfile import PlyData, PlyElement
        xyz = self.gaussians._xyz[selected_indices].detach().cpu().numpy()
        f_dc = self.gaussians._features_dc[selected_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.gaussians._features_rest[selected_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.gaussians._opacity[selected_indices].detach().cpu().numpy()
        scale = self.gaussians._scaling[selected_indices].detach().cpu().numpy()
        rotation = self.gaussians._rotation[selected_indices].detach().cpu().numpy()

        normals = np.zeros_like(xyz)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 构造属性名
        names = ['x','y','z','nx','ny','nz']
        names += [f'f_dc_{i}' for i in range(f_dc.shape[1])]
        names += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
        names += ['opacity']
        names += [f'scale_{i}' for i in range(scale.shape[1])]
        names += [f'rot_{i}' for i in range(rotation.shape[1])]
        dtype_full = [(n,'f4') for n in names]
        elements = np.empty(attributes.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(output_path)
    
    def transform_selected(self, transform_matrix: np.ndarray):
        """
        变换选中的splats
        
        Args:
            transform_matrix: 4x4变换矩阵
        """
        selected_indices = self.get_selected_indices()
        
        if len(selected_indices) == 0:
            return
        
        # 应用变换到位置
        transform_matrix = np.asarray(transform_matrix, dtype=np.float32)
        xyz = self.gaussians._xyz[selected_indices].detach().cpu().numpy()
        xyz_homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
        xyz_transformed = (transform_matrix @ xyz_homogeneous.T).T[:, :3]
        
        new_xyz = torch.from_numpy(xyz_transformed).float().to(self.gaussians._xyz.device)
        with torch.no_grad():
            self.gaussians._xyz.data[selected_indices] = new_xyz
        # 若有缓存可在此更新

    def translate_selected(self, delta: np.ndarray):
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        # 排除锁定的点
        if hasattr(self.gaussians, '_state_flags'):
            flags = self.gaussians._state_flags
            selected_flags = flags[selected_indices]
            unlocked_mask = (selected_flags & LOCK_STATE) == 0
            selected_indices = selected_indices[unlocked_mask]
        if len(selected_indices) == 0:
            return
        delta = np.asarray(delta, dtype=np.float32).reshape(1, 3)
        xyz = self.gaussians._xyz[selected_indices].detach().cpu().numpy()
        new_xyz = torch.from_numpy(xyz + delta).float().to(self.gaussians._xyz.device)
        with torch.no_grad():
            self.gaussians._xyz.data[selected_indices] = new_xyz

    def scale_selected(self, scale: np.ndarray):
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        # 排除锁定的点
        if hasattr(self.gaussians, '_state_flags'):
            flags = self.gaussians._state_flags
            selected_flags = flags[selected_indices]
            unlocked_mask = (selected_flags & LOCK_STATE) == 0
            selected_indices = selected_indices[unlocked_mask]
        if len(selected_indices) == 0:
            return
        scale = np.asarray(scale, dtype=np.float32).reshape(1, 3)
        cur = self.gaussians._scaling[selected_indices].detach().cpu().numpy()
        new_scaling = torch.from_numpy(cur + np.log(np.clip(scale, 1e-6, 1e6))).float().to(self.gaussians._scaling.device)
        with torch.no_grad():
            self.gaussians._scaling.data[selected_indices] = new_scaling

    def rotate_selected(self, rot_quat: np.ndarray):
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        # 排除锁定的点
        if hasattr(self.gaussians, '_state_flags'):
            flags = self.gaussians._state_flags
            selected_flags = flags[selected_indices]
            unlocked_mask = (selected_flags & LOCK_STATE) == 0
            selected_indices = selected_indices[unlocked_mask]
        if len(selected_indices) == 0:
            return
        rot_quat = np.asarray(rot_quat, dtype=np.float32).reshape(1, 4)
        rot_quat = rot_quat / (np.linalg.norm(rot_quat) + 1e-8)
        new_rotation = torch.from_numpy(rot_quat.repeat(selected_indices.shape[0], axis=0)).float().to(self.gaussians._rotation.device)
        with torch.no_grad():
            self.gaussians._rotation.data[selected_indices] = new_rotation

    def adjust_color_selected(self, rgb_delta: np.ndarray, brightness: float = 0.0):
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        # 排除锁定的点
        if hasattr(self.gaussians, '_state_flags'):
            flags = self.gaussians._state_flags
            selected_flags = flags[selected_indices]
            unlocked_mask = (selected_flags & LOCK_STATE) == 0
            selected_indices = selected_indices[unlocked_mask]
        if len(selected_indices) == 0:
            return
        rgb_delta = np.asarray(rgb_delta, dtype=np.float32).reshape(1, 3)
        fdc = self.gaussians._features_dc[selected_indices].detach().cpu().numpy()
        if fdc.ndim == 3:
            if fdc.shape[1] == 1:
                fdc[:, 0, :] += rgb_delta.reshape(1, 3)
                fdc[:, 0, :] += brightness
            else:
                fdc[:, :, 0] += rgb_delta.reshape(1, 3)
                fdc[:, :, 0] += brightness
        else:
            fdc[:, :3] += rgb_delta.reshape(1, 3)
            fdc[:, :3] += brightness
        new_fdc = torch.from_numpy(fdc).float().to(self.gaussians._features_dc.device)
        with torch.no_grad():
            self.gaussians._features_dc.data[selected_indices] = new_fdc
    
    def duplicate_selected(self):
        """复制选中的splats（使用撤销系统）"""
        selected_indices = self.get_selected_indices()
        
        if len(selected_indices) == 0:
            return
        
        # 确保索引在正确的设备上（可能在get_selected_indices中已经处理，但这里再确保一次）
        device = self.gaussians.get_xyz.device
        if selected_indices.device != device:
            selected_indices = selected_indices.to(device)
        
        # 使用撤销系统
        op = DuplicateSplatOp(self.gaussians, selected_indices)
        UndoRedoSystem().add(op)
    
    def hide_selected(self):
        """隐藏选中的splats"""
        op = HideSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def unhide_selected(self):
        """显示选中的splats"""
        op = UnhideSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def lock_selected(self):
        """锁定选中的splats"""
        op = LockSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def unlock_selected(self):
        """解锁选中的splats"""
        op = UnlockSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def unlock_all(self):
        """解锁所有锁定的splats"""
        op = UnlockAllSplatOp(self.gaussians)
        UndoRedoSystem().add(op)
    
    def adjust_color(self, rgb_adjustment, brightness=0.0, transparency=0.0):
        """
        调整选中splats的颜色
        
        Args:
            rgb_adjustment: [r, g, b] delta values
            brightness: 亮度调整值
            transparency: 透明度调整值
        """
        op = ColorAdjustmentOp(self.gaussians, rgb_adjustment, brightness, transparency)
        UndoRedoSystem().add(op)
    
    def paint_color(self, color, mix_weight=0.5):
        """
        绘制颜色到选中的splats
        
        Args:
            color: [r, g, b] 目标颜色
            mix_weight: 混合权重 (0-1)
        """
        op = PaintColorOp(self.gaussians, color, mix_weight)
        UndoRedoSystem().add(op)
    
    def transform_selected_with_undo(self, translation=None, rotation=None, scale=None, center=None):
        """
        使用撤销系统变换选中的splats
        
        Args:
            translation: [x, y, z] 平移向量
            rotation: [w, x, y, z] 四元数
            scale: [sx, sy, sz] 缩放因子
            center: [x, y, z] 变换中心点
        """
        op = TransformOp(self.gaussians, translation, rotation, scale, center)
        UndoRedoSystem().add(op)
    
    def get_selection_stats(self):
        """获取选择统计信息"""
        if not hasattr(self.gaussians, '_state_flags'):
            return {
                'total': 0,
                'selected': 0,
                'hidden': 0,
                'locked': 0,
                'deleted': 0
            }
        
        flags = self.gaussians._state_flags
        total = len(flags)
        selected = torch.sum((flags & SELECT_STATE) != 0).item()
        hidden = torch.sum((flags & HIDE_STATE) != 0).item()
        locked = torch.sum((flags & LOCK_STATE) != 0).item()
        deleted = torch.sum((flags & DELETE_STATE) != 0).item()
        
        return {
            'total': total,
            'selected': selected,
            'hidden': hidden,
            'locked': locked,
            'deleted': deleted,
            'visible': total - hidden - deleted
        }
    
    def filter_by_radius(self, min_radius=None, max_radius=None, op_type: EditSelectOpType = EditSelectOpType.SET):
        """
        按半径过滤选择
        
        Args:
            min_radius: 最小半径
            max_radius: 最大半径
            op_type: 选择操作类型
        """
        if not hasattr(self.gaussians, '_scaling'):
            return
        
        # 计算每个点的平均半径
        scaling = torch.exp(self.gaussians._scaling)  # 转换为实际尺寸
        avg_radius = torch.mean(scaling, dim=1)
        
        # 创建过滤mask
        mask = torch.ones(len(avg_radius), dtype=torch.bool, device=avg_radius.device)
        if min_radius is not None:
            mask &= (avg_radius >= min_radius)
        if max_radius is not None:
            mask &= (avg_radius <= max_radius)
        
        op = SplatSelectionOp(self.gaussians, op_type, precomputed_mask=mask)
        UndoRedoSystem().add(op)
    
    def filter_by_opacity(self, min_opacity=None, max_opacity=None, op_type: EditSelectOpType = EditSelectOpType.SET):
        """
        按不透明度过滤选择
        
        Args:
            min_opacity: 最小不透明度
            max_opacity: 最大不透明度
            op_type: 选择操作类型
        """
        if not hasattr(self.gaussians, '_opacity'):
            return
        
        opacity = torch.sigmoid(self.gaussians._opacity.squeeze())  # 转换为0-1范围
        
        # 创建过滤mask
        mask = torch.ones(len(opacity), dtype=torch.bool, device=opacity.device)
        if min_opacity is not None:
            mask &= (opacity >= min_opacity)
        if max_opacity is not None:
            mask &= (opacity <= max_opacity)
        
        op = SplatSelectionOp(self.gaussians, op_type, precomputed_mask=mask)
        UndoRedoSystem().add(op)
    
    def get_selection_center(self):
        """获取选中点的中心位置"""
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return None
        
        selected_xyz = self.gaussians._xyz[selected_indices]
        center = torch.mean(selected_xyz, dim=0)
        return center.cpu().numpy()
    
    def get_selection_bounds(self):
        """获取选中点的包围盒"""
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return None
        
        selected_xyz = self.gaussians._xyz[selected_indices]
        min_point = torch.min(selected_xyz, dim=0)[0].cpu().numpy()
        max_point = torch.max(selected_xyz, dim=0)[0].cpu().numpy()
        
        return BoundingBox(min_point=min_point, max_point=max_point)
    
    # ==================== Pivot 和对齐功能 ====================
    
    def set_pivot_to_selection_center(self):
        """将pivot点设置到选中点的中心"""
        center = self.get_selection_center()
        if center is not None:
            self.pivot.set_position(center)
            return center
        return None
    
    def set_pivot_to_world_origin(self):
        """将pivot点设置到世界原点"""
        self.pivot.set_position(np.array([0.0, 0.0, 0.0]))
    
    def set_pivot_to_position(self, position: np.ndarray):
        """将pivot点设置到指定位置"""
        self.pivot.set_position(position)
    
    def get_pivot_position(self) -> np.ndarray:
        """获取pivot点位置"""
        return self.pivot.get_position()
    
    def align_selection_to_axis(self, axis: str = 'x'):
        """
        将选中点对齐到坐标轴
        
        Args:
            axis: 'x', 'y', 或 'z'
        """
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        
        device = self.gaussians.get_xyz.device
        selected_xyz = self.gaussians._xyz[selected_indices]
        
        # 获取pivot位置
        pivot_pos = torch.from_numpy(self.pivot.get_position()).float().to(device)
        
        # 根据轴向对齐
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        with torch.no_grad():
            # 将选中点在指定轴上的坐标设置为pivot的坐标
            self.gaussians._xyz.data[selected_indices, axis_idx] = pivot_pos[axis_idx]
    
    def align_selection_to_grid(self, grid_size: float = 0.1):
        """
        将选中点对齐到网格
        
        Args:
            grid_size: 网格大小
        """
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        
        device = self.gaussians.get_xyz.device
        selected_xyz = self.gaussians._xyz[selected_indices]
        
        with torch.no_grad():
            # 对齐到最近的网格点
            aligned_xyz = torch.round(selected_xyz / grid_size) * grid_size
            self.gaussians._xyz.data[selected_indices] = aligned_xyz
    
    def distribute_selection_evenly(self, axis: str = 'x', spacing: float = None):
        """
        沿指定轴均匀分布选中的点
        
        Args:
            axis: 'x', 'y', 或 'z'
            spacing: 间距，如果为None则自动计算
        """
        selected_indices = self.get_selected_indices()
        if len(selected_indices) <= 1:
            return
        
        device = self.gaussians.get_xyz.device
        selected_xyz = self.gaussians._xyz[selected_indices].clone()
        
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        # 按轴坐标排序
        sorted_indices = torch.argsort(selected_xyz[:, axis_idx])
        
        # 计算起始和结束位置
        start_pos = selected_xyz[sorted_indices[0], axis_idx].item()
        end_pos = selected_xyz[sorted_indices[-1], axis_idx].item()
        
        # 计算间距
        if spacing is None:
            spacing = (end_pos - start_pos) / (len(selected_indices) - 1)
        
        # 均匀分布
        with torch.no_grad():
            for i, idx in enumerate(sorted_indices):
                new_pos = start_pos + i * spacing
                self.gaussians._xyz.data[selected_indices[idx], axis_idx] = new_pos
    
    def mirror_selection(self, axis: str = 'x', mirror_plane_offset: float = 0.0):
        """
        镜像选中的点
        
        Args:
            axis: 镜像轴 'x', 'y', 或 'z'
            mirror_plane_offset: 镜像平面的偏移
        """
        selected_indices = self.get_selected_indices()
        if len(selected_indices) == 0:
            return
        
        device = self.gaussians.get_xyz.device
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        with torch.no_grad():
            # 计算镜像位置
            xyz = self.gaussians._xyz[selected_indices].clone()
            distance_to_plane = xyz[:, axis_idx] - mirror_plane_offset
            xyz[:, axis_idx] = mirror_plane_offset - distance_to_plane
            
            # 创建镜像的点（复制）
            new_xyz = xyz
            new_features_dc = self.gaussians._features_dc[selected_indices].clone()
            new_features_rest = self.gaussians._features_rest[selected_indices].clone()
            new_scaling = self.gaussians._scaling[selected_indices].clone()
            new_rotation = self.gaussians._rotation[selected_indices].clone()
            new_opacity = self.gaussians._opacity[selected_indices].clone()
            
            # 添加到模型
            self.gaussians._xyz = torch.cat([self.gaussians._xyz, new_xyz], dim=0)
            self.gaussians._features_dc = torch.cat([self.gaussians._features_dc, new_features_dc], dim=0)
            self.gaussians._features_rest = torch.cat([self.gaussians._features_rest, new_features_rest], dim=0)
            self.gaussians._scaling = torch.cat([self.gaussians._scaling, new_scaling], dim=0)
            self.gaussians._rotation = torch.cat([self.gaussians._rotation, new_rotation], dim=0)
            self.gaussians._opacity = torch.cat([self.gaussians._opacity, new_opacity], dim=0)
            
            # 更新状态标志
            if hasattr(self.gaussians, '_state_flags'):
                new_flags = torch.zeros(len(selected_indices), dtype=torch.uint8, device=device)
                self.gaussians._state_flags = torch.cat([self.gaussians._state_flags, new_flags], dim=0)

