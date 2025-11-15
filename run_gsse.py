import os
import argparse
import gc
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
import cv2
import torch.nn.functional as nnF

# 设置CUDA内存管理优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# GSSE utilities
from seg_utils import (
    conv2d_matrix,
    compute_ratios,
    update,
    grounding_dino_prompt,
)

# 3DGS imports
from gaussiansplatting.arguments import ModelParams, PipelineParams
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.gaussian_renderer import render


# -----------------------------
# Memory Management Utilities
# -----------------------------
def clear_gpu_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3     # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return f"已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB, 总计: {total:.2f}GB"

def optimize_memory_settings():
    """优化内存设置"""
    if torch.cuda.is_available():
        # 设置内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
        # 启用内存池
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# -----------------------------
# Helpers ported from notebook
# -----------------------------
def get_3d_prompts(prompts_2d: torch.Tensor, point_image: torch.Tensor, xyz: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
    r = 4
    x_range = torch.arange(prompts_2d[0] - r, prompts_2d[0] + r, device=point_image.device)
    y_range = torch.arange(prompts_2d[1] - r, prompts_2d[1] + r, device=point_image.device)
    x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='xy')
    neighbors = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2)
    prompts_index = [torch.where((point_image == p).all(dim=1))[0] for p in neighbors]
    indexs: List[torch.Tensor] = []
    for index in prompts_index:
        if index.nelement() > 0:
            indexs.append(index)
    
    # 检查是否有有效的索引
    if len(indexs) == 0:
        # 如果没有找到匹配的点，返回一个默认的3D点
        print("警告: 在指定区域未找到匹配的3D点，使用默认点")
        return xyz[:1]  # 返回第一个点作为默认值
    
    indexs = torch.unique(torch.cat(indexs, dim=0))
    indexs_depth = depth[indexs]
    valid_depth = indexs_depth[indexs_depth > 0]
    _, sorted_indices = torch.sort(valid_depth)
    valid_indexs = indexs[depth[indexs] > 0][sorted_indices[0]]
    return xyz[valid_indexs][:3].unsqueeze(0)


def generate_3d_prompts(xyz: torch.Tensor, viewpoint_camera, prompts_2d: np.ndarray) -> torch.Tensor:
    w2c_matrix = viewpoint_camera.world_view_transform
    full_matrix = viewpoint_camera.full_proj_transform
    
    # 确保所有张量在同一设备上
    target_device = xyz.device
    w2c_matrix = w2c_matrix.to(target_device)
    full_matrix = full_matrix.to(target_device)
    
    xyz_h = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_hom = (xyz_h @ full_matrix).transpose(0, 1)
    p_w = 1.0 / (p_hom[-1, :] + 1e-7)
    p_proj = p_hom[:3, :] * p_w
    p_view = (xyz_h @ w2c_matrix[:, :3]).transpose(0, 1)
    depth = p_view[-1, :].detach().clone()

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h], device=target_device).unsqueeze(-1) - 1)
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1)).long()

    prompts_2d_t = torch.tensor(prompts_2d, device=target_device)
    prompts_3d: List[torch.Tensor] = []
    for i in range(prompts_2d_t.shape[0]):
        prompts_3d.append(get_3d_prompts(prompts_2d_t[i], point_image, xyz_h, depth))
    prompts_3D = torch.cat(prompts_3d, dim=0)
    return prompts_3D


def porject_to_2d(viewpoint_camera, points3D: torch.Tensor) -> torch.Tensor:
    full_matrix = viewpoint_camera.full_proj_transform
    
    # 确保所有张量在同一设备上
    target_device = points3D.device
    full_matrix = full_matrix.to(target_device)
    
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)
    p_w = 1.0 / (p_hom[-1, :] + 1e-7)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width
    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h], device=target_device).unsqueeze(-1) - 1)
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))
    return point_image


def _resize_mask_torch(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # mask: (H, W) long/bool -> resize to (h, w) using nearest neighbor
    mask_f = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1x1xH xW
    mask_r = nnF.interpolate(mask_f, size=(h, w), mode='nearest')
    return mask_r.squeeze(0).squeeze(0).to(dtype=mask.dtype)


def post_process_mask(mask: np.ndarray, morph_kernel_size: int = 3, smooth: bool = True) -> np.ndarray:
    """
    后处理mask：形态学操作和边界平滑
    
    Args:
        mask: 输入的mask（numpy数组，0-1或0-255）
        morph_kernel_size: 形态学操作的核大小
        smooth: 是否进行平滑处理
    
    Returns:
        处理后的mask
    """
    import cv2
    
    # 确保mask是二值图像
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    # 形态学闭运算：填补小洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 形态学开运算：去除小噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 可选：平滑边界
    if smooth:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 127).astype(np.uint8) * 255
    
    return mask


def mask_inverse(xyz: torch.Tensor, viewpoint_camera, sam_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w2c_matrix = viewpoint_camera.world_view_transform
    
    # 确保所有张量在同一设备上
    target_device = xyz.device
    w2c_matrix = w2c_matrix.to(target_device)
    
    xyz_h = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz_h @ w2c_matrix[:, :3]).transpose(0, 1)
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = _resize_mask_torch(sam_mask, h, w).long()
    else:
        sam_mask = sam_mask.long()

    point_image = porject_to_2d(viewpoint_camera, xyz_h).long()
    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth
    point_mask = torch.full((point_image.shape[0],), -1, device=target_device)
    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]
    indices_mask = torch.where(point_mask == 1)[0]
    return point_mask, indices_mask


def ensemble(multiview_masks: List[torch.Tensor], threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
    multiview_masks_cat = torch.cat(multiview_masks, dim=1)
    vote_labels, _ = torch.mode(multiview_masks_cat, dim=1)
    matches = torch.eq(multiview_masks_cat, vote_labels.unsqueeze(1))
    ratios = torch.sum(matches, dim=1) / multiview_masks_cat.shape[1]
    ratios_mask = ratios > threshold
    labels_mask = (vote_labels == 1) & ratios_mask
    indices_mask = torch.where(labels_mask)[0].detach().cpu()
    return vote_labels, indices_mask


def gaussian_decomp(gaussians: GaussianModel, viewpoint_camera, input_mask: torch.Tensor, indices_mask: torch.Tensor) -> GaussianModel:
    xyz = gaussians.get_xyz
    point_image = porject_to_2d(viewpoint_camera, xyz)
    
    # 确保所有张量在同一设备上
    target_device = xyz.device
    input_mask = input_mask.to(target_device)
    indices_mask = indices_mask.to(target_device)
    
    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device=xyz.device.type)
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)
    decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)
    return decomp_gaussians


def save_gs(pc: GaussianModel, indices_mask: torch.Tensor, save_path: str) -> None:
    from plyfile import PlyData, PlyElement

    # 检查indices_mask是否为空
    if isinstance(indices_mask, torch.Tensor):
        if indices_mask.numel() == 0:
            raise ValueError(f"分割后的点云为空，无法保存PLY文件: {save_path}")
        indices_mask_np = indices_mask.cpu().numpy() if indices_mask.is_cuda else indices_mask.numpy()
    else:
        indices_mask_np = np.array(indices_mask)
        if len(indices_mask_np) == 0:
            raise ValueError(f"分割后的点云为空，无法保存PLY文件: {save_path}")
    
    # 判断是布尔掩码还是索引数组
    total_points = pc._xyz.shape[0]
    is_bool_mask = indices_mask_np.dtype == bool
    
    if is_bool_mask:
        # 布尔掩码：验证大小是否匹配
        if len(indices_mask_np) != total_points:
            raise ValueError(f"布尔掩码大小不匹配: mask_size={len(indices_mask_np)}, total_points={total_points}")
        # 检查是否有选中的点
        if not np.any(indices_mask_np):
            raise ValueError(f"布尔掩码中没有选中任何点，无法保存PLY文件: {save_path}")
    else:
        # 索引数组：验证索引范围
        if np.max(indices_mask_np) >= total_points:
            raise ValueError(f"索引超出范围: max_index={np.max(indices_mask_np)}, total_points={total_points}")
        if np.min(indices_mask_np) < 0:
            raise ValueError(f"索引为负数: min_index={np.min(indices_mask_np)}")

    xyz = pc._xyz.detach().cpu()[indices_mask_np].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask_np].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask_np].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask_np].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask_np].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask_np].numpy()

    # 再次检查提取的点数
    if xyz.shape[0] == 0:
        raise ValueError(f"提取的点云为空，无法保存PLY文件: {save_path}")

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    PlyData([el]).write(save_path)


# -----------------------------
# SAM setup (same as notebook)
# -----------------------------
def init_sam_predictor(sam_ckpt_path: str, device: str = 'cuda', model_type: str = 'vit_h'):
    # Try regular import; if missing, add local repo paths then import
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ModuleNotFoundError:
        import sys
        repo_root = os.path.dirname(__file__)
        candidate_paths = [
            os.path.join(repo_root, 'dependencies', 'sam_ckpt', 'segment-anything'),
            os.path.join(repo_root, 'gaussiansplatting', 'dependencies', 'sam_ckpt', 'segment-anything'),
        ]
        for p in candidate_paths:
            if os.path.isdir(p):
                sys.path.insert(0, p)
        from segment_anything import SamPredictor, sam_model_registry
    
    # Normalize and validate device
    device = device if (device in ['cuda', 'cpu']) else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 清理内存后尝试加载
        clear_gpu_memory()
        sam = sam_model_registry[model_type](checkpoint=sam_ckpt_path).to(device)
        predictor = SamPredictor(sam)
        print(f"SAM模型成功加载到 {device}")
        return predictor
    except torch.cuda.OutOfMemoryError:
        print(f"GPU内存不足，尝试CPU加载...")
        clear_gpu_memory()
        sam = sam_model_registry[model_type](checkpoint=sam_ckpt_path).to('cpu')
        predictor = SamPredictor(sam)
        print("SAM模型已切换到CPU模式")
        return predictor


def init_fastsam_predictor(fastsam_ckpt_path: str, device: str = 'cuda', model_type: str = 'fastsam_s'):
    """Initialize FastSAM predictor"""
    try:
        # Try to import FastSAM from ultralytics
        from ultralytics import FastSAM
    except ImportError:
        # If ultralytics not available, try to add local FastSAM path
        import sys
        repo_root = os.path.dirname(__file__)
        fastsam_path = os.path.join(repo_root, 'dependencies', 'sam_ckpt', 'FastSAM')
        if os.path.isdir(fastsam_path):
            sys.path.insert(0, fastsam_path)
            from fastsam import FastSAM
        else:
            raise ImportError("FastSAM not found. Please install ultralytics or ensure FastSAM is in dependencies/sam_ckpt/FastSAM")
    
    # Normalize and validate device
    device = device if (device in ['cuda', 'cpu']) else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 清理内存后尝试加载
        clear_gpu_memory()
        model = FastSAM(fastsam_ckpt_path)
        model.to(device)
        print(f"FastSAM模型成功加载到 {device}")
        return model
    except torch.cuda.OutOfMemoryError:
        print(f"GPU内存不足，尝试CPU加载...")
        clear_gpu_memory()
        model = FastSAM(fastsam_ckpt_path)
        model.to('cpu')
        print("FastSAM模型已切换到CPU模式")
        return model


def text_prompting(predictor, image: np.ndarray, text: str, mask_id: int):
    input_boxes = grounding_dino_prompt(image, text)
    # Ensure boxes are on the same device as predictor
    pred_device = getattr(predictor, 'device', next(predictor.model.parameters()).device)
    boxes = torch.tensor(input_boxes)[0:1].to(pred_device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    masks_np = masks[0].cpu().numpy()
    return_mask = (masks_np[mask_id, :, :, None] * 255).astype(np.uint8)
    return return_mask / 255


def self_prompt(predictor, point_prompts: torch.Tensor, sam_feature, mask_id: int):
    import numpy as np
    input_point = point_prompts.detach().cpu().numpy()
    input_label = np.ones(len(input_point))
    predictor.features = sam_feature
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return_mask = (masks[mask_id, :, :, None] * 255).astype(np.uint8)
    return return_mask / 255


def fastsam_point_prompt(fastsam_model, image: np.ndarray, point_prompts: torch.Tensor, mask_id: int, conf=0.5, iou=0.7):
    """FastSAM point prompting function
    
    Args:
        conf: 置信度阈值，默认0.5（提高精度，减少误检）
        iou: IoU阈值，默认0.7（提高精度，减少重叠）
    """
    import numpy as np
    from PIL import Image
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Convert point prompts to numpy
    input_points = point_prompts.detach().cpu().numpy()
    input_labels = np.ones(len(input_points))
    
    # Run FastSAM prediction with improved parameters
    results = fastsam_model(
        pil_image,
        device=fastsam_model.device,
        retina_masks=True,
        imgsz=1024,
        conf=conf,  # 提高置信度阈值，减少误检
        iou=iou,   # 降低IoU阈值，减少重叠mask
        max_det=100,
    )
    
    if len(results) == 0 or len(results[0].masks.data) == 0:
        # Return empty mask if no results
        h, w = image.shape[:2]
        return np.zeros((h, w, 1), dtype=np.float32)
    
    # Get the best mask based on point prompts
    masks = results[0].masks.data.cpu().numpy()
    
    # Simple selection: choose the mask that contains the most points
    best_mask_idx = 0
    max_points_in_mask = 0
    
    for i, mask in enumerate(masks):
        points_in_mask = 0
        for point in input_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0.5:
                points_in_mask += 1
        
        if points_in_mask > max_points_in_mask:
            max_points_in_mask = points_in_mask
            best_mask_idx = i
    
    # Select mask and convert to proper format
    selected_mask = masks[best_mask_idx]
    
    # Resize mask to match input image size if needed
    if selected_mask.shape != image.shape[:2]:
        import cv2
        selected_mask = cv2.resize(selected_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return_mask = (selected_mask[:, :, None] * 255).astype(np.uint8)
    return return_mask / 255


def fastsam_text_prompt(fastsam_model, image: np.ndarray, text: str, mask_id: int, conf=0.5, iou=0.7):
    """FastSAM text prompting function using CLIP
    
    Args:
        conf: 置信度阈值，默认0.5（提高精度，减少误检）
        iou: IoU阈值，默认0.7（提高精度，减少重叠）
    """
    import numpy as np
    from PIL import Image
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Run FastSAM prediction with improved parameters
    results = fastsam_model(
        pil_image,
        device=fastsam_model.device,
        retina_masks=True,
        imgsz=1024,
        conf=conf,  # 提高置信度阈值，减少误检
        iou=iou,   # 降低IoU阈值，减少重叠mask
        max_det=100,
    )
    
    if len(results) == 0 or len(results[0].masks.data) == 0:
        # Return empty mask if no results
        h, w = image.shape[:2]
        return np.zeros((h, w, 1), dtype=np.float32)
    
    # Use CLIP to select the best mask based on text prompt
    try:
        import clip
        import torch
        
        # Load CLIP model
        clip_model, preprocess = clip.load('ViT-B/32', device=fastsam_model.device)
        
        # Get masks and format them
        masks = results[0].masks.data.cpu().numpy()
        
        # Crop images for each mask and score them
        best_mask_idx = 0
        best_score = -1
        
        for i, mask in enumerate(masks):
            # Create cropped image for this mask
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Find bounding box
            coords = np.where(mask_binary)
            if len(coords[0]) == 0:
                continue
                
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Crop the image
            cropped_img = image[y_min:y_max+1, x_min:x_max+1]
            if cropped_img.size == 0:
                continue
                
            cropped_pil = Image.fromarray(cropped_img)
            
            # Preprocess and encode
            image_input = preprocess(cropped_pil).unsqueeze(0).to(fastsam_model.device)
            text_input = clip.tokenize([text]).to(fastsam_model.device)
            
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_input)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features).item()
                
                if similarity > best_score:
                    best_score = similarity
                    best_mask_idx = i
        
        # Select the best mask
        selected_mask = masks[best_mask_idx]
        
    except ImportError:
        # Fallback: just select the largest mask
        masks = results[0].masks.data.cpu().numpy()
        areas = [np.sum(mask > 0.5) for mask in masks]
        best_mask_idx = np.argmax(areas)
        selected_mask = masks[best_mask_idx]
    
    # Resize mask to match input image size if needed
    if selected_mask.shape != image.shape[:2]:
        import cv2
        selected_mask = cv2.resize(selected_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return_mask = (selected_mask[:, :, None] * 255).astype(np.uint8)
    return return_mask / 255


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run segmentation without Jupyter (supports SAGS and other methods)")
    parser.add_argument('--model_path', type=str, required=True, help='3DGS training output directory')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration to load (e.g., 7000)')
    parser.add_argument('--prompt_type', type=str, default='point', choices=['point', 'text'], help='Prompt mode')
    parser.add_argument('--point', type=str, default=None, help='Point prompt as x,y (e.g., 300,180)')
    parser.add_argument('--text', type=str, default=None, help='Text prompt (for GroundingDINO)')
    parser.add_argument('--mask_id', type=int, default=2, help='Choose 0/1/2 SAM candidate mask')
    parser.add_argument('--threshold', type=float, default=0.7, help='Multi-view voting threshold')
    parser.add_argument('--gd_interval', type=int, default=4, help='Gaussian Decomposition interval; -1 to disable')
    parser.add_argument('--sam_model', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b', 'fastsam_s', 'fastsam_x'], help='SAM model type')
    parser.add_argument('--sam_ckpt', type=str, default=None, help='Path to SAM checkpoint .pth (overrides --sam_model)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: point_cloud/iteration_X)')
    parser.add_argument('--render_obj', action='store_true', help='Render segmented object views to images')
    parser.add_argument('--source_path', type=str, default=None, help='Override source dataset path (else read from cfg_args)')
    parser.add_argument('--resolution', type=int, default=None, help='Downscale factor {1,2,4,8} or -1 for auto')
    parser.add_argument('--sam_long_side', type=int, default=640, help='Resize SAM input long side (e.g., 640/768/1024)')
    args = parser.parse_args()

    # 初始化内存管理
    optimize_memory_settings()
    print(f"初始内存状态: {get_gpu_memory_info()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup dataset / scene
    # Resolve source_path via cfg_args if not provided
    cfg_args_path = os.path.join(args.model_path, 'cfg_args')
    cfg_ns = None
    if args.source_path is None and os.path.isfile(cfg_args_path):
        try:
            with open(cfg_args_path, 'r') as f:
                cfg_text = f.read().strip()
            cfg_ns = eval(cfg_text)
        except Exception:
            cfg_ns = None

    resolved_source_path = args.source_path or (getattr(cfg_ns, 'source_path', None) if cfg_ns else None)
    if not resolved_source_path:
        raise ValueError('source_path is required but not found. Provide --source_path or ensure cfg_args exists in model_path.')

    # Build a minimal args namespace compatible with Scene and GaussianModel
    scene_args = argparse.Namespace(
        model_path=args.model_path,
        source_path=resolved_source_path,
        images=getattr(cfg_ns, 'images', 'images') if cfg_ns else 'images',
        white_background=getattr(cfg_ns, 'white_background', False) if cfg_ns else False,
        eval=False,
        sh_degree=getattr(cfg_ns, 'sh_degree', 3) if cfg_ns else 3,
        resolution=(args.resolution if args.resolution is not None else (getattr(cfg_ns, 'resolution', -1) if cfg_ns else -1)),
        data_device=getattr(cfg_ns, 'data_device', 'cuda') if cfg_ns else 'cuda',
    )
    pipeline = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    gaussians = GaussianModel(scene_args.sh_degree)
    scene = Scene(scene_args, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    bg_color = [1, 1, 1] if scene_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Output paths
    out_dir = args.out_dir or os.path.join(args.model_path, f"point_cloud/iteration_{args.iteration}")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'point_cloud_seg.ply')
    save_gd_path = os.path.join(out_dir, 'point_cloud_seg_gd.ply')

    # SAM setup
    if args.sam_ckpt:
        # 用户指定了模型路径
        sam_ckpt_path = args.sam_ckpt
        model_type = 'vit_h'  # 默认类型，实际会根据文件自动检测
        print(f"使用用户指定的SAM模型: {sam_ckpt_path}")
    else:
        # 根据选择的模型类型确定权重文件路径
        sam_model_paths = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth', 
            'vit_b': 'sam_vit_b_01ec64.pth',
            'fastsam_s': 'FastSAM-s.pt',
            'fastsam_x': 'FastSAM-x.pt'
        }
        
        sam_ckpt_filename = sam_model_paths.get(args.sam_model, 'sam_vit_h_4b8939.pth')
        sam_ckpt_path = os.path.join(os.path.dirname(__file__), 'dependencies', 'sam_ckpt', sam_ckpt_filename)
        model_type = args.sam_model
        print(f"使用SAM {args.sam_model.upper()}模型: {sam_ckpt_filename}")
    
    if not os.path.isfile(sam_ckpt_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {sam_ckpt_path}")
    
    # Initialize predictor based on model type
    if model_type.startswith('fastsam'):
        predictor = init_fastsam_predictor(sam_ckpt_path, model_type=model_type)
    else:
        predictor = init_sam_predictor(sam_ckpt_path, model_type=model_type)

    # Pre-extract SAM features per view
    sam_features = {}
    print('Preprocessing: extracting SAM features...')
    print(f"SAM预处理前内存状态: {get_gpu_memory_info()}")
    
    for i, view in enumerate(cameras):
        image_name = view.image_name
        print(f"处理视图 {i+1}/{len(cameras)}: {image_name}")
        
        # 定期清理内存
        if i % 5 == 0:
            clear_gpu_memory()
        
        with torch.no_grad():
            render_pkg = render(view, gaussians, pipeline, background)
            render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
        
        # Resize for SAM to reduce memory
        h, w = render_image.shape[:2]
        long_side = max(h, w)
        if long_side > args.sam_long_side:
            scale = args.sam_long_side / long_side
            new_w, new_h = int(w * scale), int(h * scale)
            render_image = cv2.resize(render_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Handle FastSAM vs SAM differently
        if model_type.startswith('fastsam'):
            # FastSAM doesn't need feature extraction - we'll run inference directly
            sam_features[image_name] = render_image  # Store the image for later use
        else:
            try:
                predictor.set_image(render_image)
            except torch.cuda.OutOfMemoryError:
                print("GPU内存不足，清理内存后重试...")
                clear_gpu_memory()
                predictor.set_image(render_image)
            
            # Move features to CPU to free GPU memory; will move back when using
            feats = predictor.features
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu()
            sam_features[image_name] = feats
            del feats
        
        # 每处理几个视图就清理一次内存
        if i % 3 == 0:
            clear_gpu_memory()
    
    print(f"SAM预处理完成，最终内存状态: {get_gpu_memory_info()}")

    # Prepare prompts
    xyz = gaussians.get_xyz
    if args.prompt_type == 'point':
        if not args.point:
            raise ValueError('--point is required for prompt_type=point (format: x,y)')
        x_str, y_str = args.point.split(',')
        input_point = np.asarray([[int(x_str), int(y_str)]], dtype=np.int32)
        # first view for 3D prompts
        first_view = cameras[0]
        prompts_3d = generate_3d_prompts(xyz, first_view, input_point.tolist())
    else:
        if not args.text:
            raise ValueError('--text is required for prompt_type=text')
        input_point = None
        prompts_3d = None

    # Iterate views -> masks -> label assignment
    multiview_masks: List[torch.Tensor] = []
    sam_masks: List[torch.Tensor] = []
    print(f"开始分割处理，当前内存状态: {get_gpu_memory_info()}")
    
    for i, view in enumerate(cameras):
        image_name = view.image_name
        print(f"处理视图 {i+1}/{len(cameras)}: {image_name}")
        
        # 定期清理内存
        if i % 3 == 0:
            clear_gpu_memory()
        
        with torch.no_grad():
            render_pkg = render(view, gaussians, pipeline, background)
            render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

        try:
            if model_type.startswith('fastsam'):
                # FastSAM prediction
                if args.prompt_type == 'point':
                    prompts_2d = porject_to_2d(view, prompts_3d)
                    sam_mask_np = fastsam_point_prompt(predictor, render_image, prompts_2d, args.mask_id)
                else:
                    sam_mask_np = fastsam_text_prompt(predictor, render_image, args.text, args.mask_id)
            else:
                # Regular SAM prediction
                if args.prompt_type == 'point':
                    prompts_2d = porject_to_2d(view, prompts_3d)
                    # Ensure features on CUDA before predict
                    feats = sam_features[image_name]
                    pred_device = getattr(predictor, 'device', next(predictor.model.parameters()).device)
                    if isinstance(feats, torch.Tensor) and feats.device.type != pred_device.type:
                        feats = feats.to(pred_device, non_blocking=True)
                    sam_mask_np = self_prompt(predictor, prompts_2d, feats, args.mask_id)
                else:
                    sam_mask_np = text_prompting(predictor, render_image, args.text, args.mask_id)
        except torch.cuda.OutOfMemoryError:
            print("GPU内存不足，清理内存后重试...")
            clear_gpu_memory()
            if model_type.startswith('fastsam'):
                # FastSAM prediction
                if args.prompt_type == 'point':
                    prompts_2d = porject_to_2d(view, prompts_3d)
                    sam_mask_np = fastsam_point_prompt(predictor, render_image, prompts_2d, args.mask_id)
                else:
                    sam_mask_np = fastsam_text_prompt(predictor, render_image, args.text, args.mask_id)
            else:
                # Regular SAM prediction
                if args.prompt_type == 'point':
                    prompts_2d = porject_to_2d(view, prompts_3d)
                    feats = sam_features[image_name]
                    pred_device = getattr(predictor, 'device', next(predictor.model.parameters()).device)
                    if isinstance(feats, torch.Tensor) and feats.device.type != pred_device.type:
                        feats = feats.to(pred_device, non_blocking=True)
                    sam_mask_np = self_prompt(predictor, prompts_2d, feats, args.mask_id)
                else:
                    sam_mask_np = text_prompting(predictor, render_image, args.text, args.mask_id)

        sam_mask = torch.from_numpy(sam_mask_np).to(device)
        if len(sam_mask.shape) != 2:
            sam_mask = sam_mask.squeeze(-1)
        sam_mask = sam_mask.long()
        sam_masks.append(sam_mask)

        # 确保xyz和sam_mask在同一设备上
        if device.type == 'cpu':
            xyz_cpu = xyz.cpu()
            point_mask, indices_mask = mask_inverse(xyz_cpu, view, sam_mask)
        else:
            point_mask, indices_mask = mask_inverse(xyz, view, sam_mask)
        multiview_masks.append(point_mask.unsqueeze(-1))
    
    print(f"分割处理完成，最终内存状态: {get_gpu_memory_info()}")

    # Multi-view voting - 根据模型类型调整阈值
    if args.sam_model == 'vit_b':
        # ViT-B模型精度较低，使用更低的阈值
        adjusted_threshold = max(0.3, args.threshold - 0.2)
        print(f"ViT-B模型检测，调整投票阈值: {args.threshold} -> {adjusted_threshold}")
    elif args.sam_model == 'vit_l':
        # ViT-L模型中等精度，稍微降低阈值
        adjusted_threshold = max(0.4, args.threshold - 0.1)
        print(f"ViT-L模型检测，调整投票阈值: {args.threshold} -> {adjusted_threshold}")
    elif args.sam_model.startswith('fastsam'):
        # FastSAM模型，使用中等阈值
        adjusted_threshold = max(0.4, args.threshold - 0.1)
        print(f"FastSAM模型检测，调整投票阈值: {args.threshold} -> {adjusted_threshold}")
    else:
        # ViT-H模型高精度，使用原始阈值
        adjusted_threshold = args.threshold
    
    _, final_mask = ensemble(multiview_masks, threshold=adjusted_threshold)

    # Save before GD
    print(f'Saving segmented PLY (no GD): {save_path}')
    save_gs(gaussians, final_mask, save_path)

    # Optional GD as post-process
    if args.gd_interval != -1:
        print('Applying Gaussian Decomposition (post-process)...')
        for i, view in enumerate(cameras):
            if i % args.gd_interval == 0:
                input_mask = sam_masks[i]
                gaussians = gaussian_decomp(gaussians, view, input_mask, final_mask.to(device))
        print(f'Saving segmented PLY (with GD): {save_gd_path}')
        save_gs(gaussians, final_mask, save_gd_path)

    # Optional rendering
    if args.render_obj:
        seg_ply = save_gd_path if os.path.isfile(save_gd_path) else save_path
        print(f'Rendering segmented object views from {seg_ply} ...')
        seg_gaussians = GaussianModel(scene_args.sh_degree)
        seg_gaussians.load_ply(seg_ply)
        obj_dir = os.path.join(args.model_path, 'obj_images')
        os.makedirs(obj_dir, exist_ok=True)
        for view in cameras:
            with torch.no_grad():
                rendering = render(view, seg_gaussians, pipeline, background)["render"]  # CHW, [0,1]
            img = (rendering.clamp(0,1).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)  # HWC RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(obj_dir, f"{view.image_name}.png"), img_bgr)


if __name__ == '__main__':
    main()


