#!/usr/bin/env python3
"""
SegAnyGAussians (SAGA) 模块
封装SAGA项目的所有核心功能，用于集成到GUI中
"""

import os
import sys
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import traceback

# 添加SegAnyGAussians到路径
SAGA_ROOT = os.path.join(os.path.dirname(__file__), "dependencies", "SegAnyGAussians")
if SAGA_ROOT not in sys.path:
    sys.path.insert(0, SAGA_ROOT)

# 添加子模块路径以便导入编译后的模块
SAGA_SUBMODULES = os.path.join(SAGA_ROOT, "submodules")
# 添加各个子模块的路径
for submodule in ["diff-gaussian-rasterization", "diff-gaussian-rasterization_contrastive_f", 
                  "diff-gaussian-rasterization-depth", "simple-knn"]:
    submodule_path = os.path.join(SAGA_SUBMODULES, submodule)
    if os.path.exists(submodule_path) and submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)

# 尝试导入SAGA依赖，如果导入失败，模块仍然可以创建但功能受限
_IMPORT_ERRORS = []
try:
    from scene import Scene, GaussianModel, FeatureGaussianModel
except ImportError as e:
    _IMPORT_ERRORS.append(str(e))
    Scene = None
    GaussianModel = None
    FeatureGaussianModel = None

try:
    from gaussian_renderer import render, render_contrastive_feature, render_mask
except ImportError as e:
    _IMPORT_ERRORS.append(f"gaussian_renderer: {str(e)}")
    render = None
    render_contrastive_feature = None
    render_mask = None
    # 如果是因为缺少diff_gaussian_rasterization_depth，尝试只导入基础render函数
    if 'diff_gaussian_rasterization_depth' in str(e):
        try:
            # 尝试只导入render函数（它不依赖depth模块）
            # 这需要临时修改gaussian_renderer/__init__.py或使用其他方法
            # 暂时标记为不可用
            pass
        except:
            pass

try:
    from arguments import ModelParams, PipelineParams, OptimizationParams
except ImportError as e:
    _IMPORT_ERRORS.append(str(e))
    ModelParams = None
    PipelineParams = None
    OptimizationParams = None

try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import cv2
except ImportError:
    cv2 = None

# 如果有导入错误，打印警告但不阻止模块创建
if _IMPORT_ERRORS:
    print(f"警告: SAGA模块部分依赖导入失败: {', '.join(_IMPORT_ERRORS[:3])}")
    print("某些功能可能不可用，请确保已安装SegAnyGAussians依赖")

class SAGAModule:
    """SAGA核心功能模块"""
    
    def __init__(self, model_path: str, source_path: str = None):
        """
        初始化SAGA模块
        
        Args:
            model_path: 3DGS模型路径
            source_path: COLMAP数据路径（如果为None，会尝试从cfg_args读取）
        """
        self.model_path = model_path
        self.source_path = source_path
        self.gaussians = None
        self.feature_gaussians = None
        self.scene = None
        self.cameras = None
        self.pipeline = None
        self.background = None
        self.iteration = None  # 当前加载的迭代次数
        
        # 分割相关
        self.current_mask = None
        self.point_prompts = []  # [(x, y, label), ...]
        self.segmentation_history = []  # 用于撤销操作
        
        # 聚类相关
        self.cluster_labels = None
        
    def load_model(self, iteration: int = -1, sh_degree: int = 3, 
                   feature_dim: int = 32, white_background: bool = False):
        """
        加载3DGS模型和场景
        
        Args:
            iteration: 迭代次数，-1表示自动查找最新
            sh_degree: 球谐函数度数
            feature_dim: 特征维度
            white_background: 是否使用白色背景
        """
        if ModelParams is None or GaussianModel is None or FeatureGaussianModel is None or Scene is None:
            raise ImportError("SAGA核心模块未正确导入，请检查依赖")
        
        try:
            # 如果没有提供source_path，尝试从cfg_args读取
            if self.source_path is None:
                cfg_path = os.path.join(self.model_path, 'cfg_args')
                if os.path.isfile(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg_ns = eval(f.read().strip())
                    self.source_path = getattr(cfg_ns, 'source_path', '')
                    white_background = getattr(cfg_ns, 'white_background', False)
            
            # 创建ArgumentParser用于ModelParams
            from argparse import ArgumentParser
            temp_parser = ArgumentParser()
            
            # 构建参数
            model_args = ModelParams(
                parser=temp_parser,
                sentinel=True
            )
            model_args.model_path = self.model_path
            model_args.source_path = self.source_path
            model_args.images = "images"
            model_args.white_background = white_background
            model_args.sh_degree = sh_degree
            model_args.feature_dim = feature_dim
            model_args.eval = False
            model_args.need_features = False
            model_args.need_masks = False
            model_args.data_device = "cuda"
            # 设置resolution属性（需要不带下划线的版本）
            model_args.resolution = -1  # 默认使用原始分辨率
            if hasattr(model_args, '_resolution'):
                model_args._resolution = -1
            
            # 初始化模型
            self.gaussians = GaussianModel(sh_degree)
            self.feature_gaussians = FeatureGaussianModel(feature_dim)
            
            # 加载场景
            self.scene = Scene(
                model_args, 
                self.gaussians, 
                self.feature_gaussians,
                load_iteration=iteration,
                shuffle=False,
                target='scene',
                mode='eval'
            )
            
            self.cameras = self.scene.getTrainCameras()
            
            # 保存迭代信息（供SIBR查看器使用）
            self.iteration = iteration if iteration > 0 else self.scene.loaded_iter
            
            # 设置渲染参数
            bg_color = [1, 1, 1] if white_background else [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            # 设置pipeline
            temp_parser2 = ArgumentParser()
            self.pipeline = PipelineParams(parser=temp_parser2)
            self.pipeline.convert_SHs_python = False
            self.pipeline.compute_cov3D_python = False
            self.pipeline.debug = False
            
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            traceback.print_exc()
            return False
    
    def load_feature_model(self, iteration: int = -1, target: str = 'contrastive_feature'):
        """
        加载特征高斯模型
        
        Args:
            iteration: 特征模型迭代次数，-1表示自动查找最新
            target: 目标类型 ('contrastive_feature' 或 'feature')
            
        Returns:
            bool: 是否成功加载
        """
        if ModelParams is None or FeatureGaussianModel is None or Scene is None:
            raise ImportError("SAGA核心模块未正确导入，请检查依赖")
        
        # 先检查特征模型是否存在
        if not self.check_feature_model_exists(iteration, target):
            print(f"\n特征模型不存在（target={target}）。")
            print("提示：特征模型需要先通过train_contrastive_features()训练生成。")
            print("请先运行特征训练，或者检查模型路径是否正确。")
            return False
        
        try:
            from argparse import ArgumentParser
            temp_parser = ArgumentParser()
            model_args = ModelParams(parser=temp_parser, sentinel=True)
            model_args.model_path = self.model_path
            model_args.source_path = self.source_path
            model_args.images = "images"
            model_args.white_background = False
            model_args.sh_degree = 3
            model_args.feature_dim = 32
            model_args.eval = True
            model_args.need_features = False
            model_args.need_masks = False
            model_args.data_device = "cuda"
            model_args.resolution = -1
            if hasattr(model_args, '_resolution'):
                model_args._resolution = -1
            
            # 重新加载特征模型
            self.feature_gaussians = FeatureGaussianModel(32)
            
            feature_scene = Scene(
                model_args,
                self.gaussians,
                self.feature_gaussians,
                load_iteration=iteration,
                shuffle=False,
                target=target,
                mode='eval'
            )
            
            print(f"特征模型加载成功 (target={target})")
            return True
        except FileNotFoundError as e:
            print(f"\n加载特征模型失败: 文件不存在")
            print(f"路径: {str(e)}")
            print("提示：特征模型需要先通过train_contrastive_features()训练生成。")
            return False
        except Exception as e:
            print(f"加载特征模型失败: {e}")
            traceback.print_exc()
            return False
    
    def check_feature_model_exists(self, iteration: int = -1, target: str = 'contrastive_feature'):
        """
        检查特征模型是否存在
        
        Args:
            iteration: 迭代次数，-1表示自动查找最新，None表示不检查
            target: 目标类型
        
        Returns:
            bool: 是否存在特征模型
        """
        # 如果iteration是None，表示不检查，返回False
        if iteration is None:
            return False
            
        try:
            from utils.system_utils import searchForMaxIteration
        except ImportError:
            return False
        
        try:
            if iteration == -1:
                feature_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud"), 
                    target=target
                )
            else:
                feature_iter = iteration
            
            if feature_iter is None:
                return False
            
            # 检查特征模型文件（支持两种命名方式）
            feature_path = os.path.join(
                self.model_path,
                "point_cloud",
                f"iteration_{feature_iter}",
                f"{target}_point_cloud.ply"
            )
            
            if os.path.exists(feature_path):
                return True
            
            # 对于contrastive_feature，如果找不到，也不返回True（因为需要训练生成）
            return False
        except:
            return False
    
    def train_contrastive_features(self, iterations: int = 10000, 
                                   num_sampled_rays: int = 1000,
                                   smooth_K: int = 16,
                                   scale_aware_dim: int = -1,
                                   feature_lr: float = 0.0025,
                                   rfn: float = 1.0,
                                   callback=None,
                                   log_callback=None):
        """
        训练对比特征
        
        Args:
            iterations: 训练迭代次数
            num_sampled_rays: 采样光线数量
            smooth_K: 平滑K值
            scale_aware_dim: 尺度感知维度
            feature_lr: 特征学习率
            rfn: RFN权重
            callback: 进度回调函数 callback(iteration, total, loss_info)
            log_callback: 日志回调函数 log_callback(message, level)
        """
        if ModelParams is None or OptimizationParams is None or PipelineParams is None:
            raise ImportError("SAGA核心模块未正确导入，请检查依赖")
        
        try:
            from argparse import ArgumentParser
            
            # 导入训练函数（这可能会因为缺少diff_gaussian_rasterization_depth而失败）
            sys.path.insert(0, SAGA_ROOT)
            try:
                from train_contrastive_feature import training, prepare_output_and_logger
            except ImportError as e:
                if 'diff_gaussian_rasterization_depth' in str(e):
                    raise ImportError("训练功能需要diff_gaussian_rasterization_depth模块，请安装SegAnyGAussians的完整依赖")
                raise
            
            # 检查必需的sam_masks数据是否存在
            sam_masks_dir = None
            if self.source_path:
                sam_masks_dir = os.path.join(self.source_path, "sam_masks")
            if not sam_masks_dir or not os.path.exists(sam_masks_dir) or not os.listdir(sam_masks_dir):
                print("\n错误：缺少必需的sam_masks数据！")
                print("训练对比特征需要先准备SAM masks数据。")
                print("\n请按照以下步骤准备数据：")
                print("1. 运行 extract_segment_everything_masks.py 提取SAM masks")
                print("   例如: python extract_segment_everything_masks.py \\")
                print("         --image_root <场景数据路径> \\")
                print("         --sam_checkpoint_path <SAM模型路径> \\")
                print("         --downsample 1")
                print("\n2. 运行 get_scale.py 获取mask scales（可选，如果scale_aware_dim=-1）")
                print("   例如: python get_scale.py \\")
                print("         --image_root <场景数据路径> \\")
                print("         --model_path <3DGS模型路径>")
                print("\n请参考SegAnyGAussians的README.md了解详细步骤。")
                raise FileNotFoundError(f"找不到sam_masks目录: {sam_masks_dir if sam_masks_dir else '未指定source_path'}")
            
            # 检查并创建mask_scales目录和文件（如果缺少）
            # 如果scale_aware_dim=-1，可以创建空的mask_scales文件
            mask_scales_dir = os.path.join(self.source_path, "mask_scales")
            images_dir = os.path.join(self.source_path, "images")
            
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))]
                
                os.makedirs(mask_scales_dir, exist_ok=True)
                
                # 检查哪些图像缺少mask_scales文件
                missing_scales = []
                for image_file in image_files:
                    name = os.path.splitext(image_file)[0]
                    scale_file = os.path.join(mask_scales_dir, name + '.pt')
                    if not os.path.exists(scale_file):
                        missing_scales.append(name)
                
                if missing_scales:
                    print(f"\n警告: 发现 {len(missing_scales)} 个图像缺少mask_scales文件")
                    print("\n重要提示: mask_scales 数据对训练至关重要！")
                    print("建议先运行 get_scale.py 生成正确的 mask_scales:")
                    print("  python dependencies/SegAnyGAussians/get_scale.py \\")
                    print(f"    --image_root {self.source_path} \\")
                    print(f"    --model_path {self.model_path}")
                    print("\n如果继续训练而不生成 mask_scales，可能会出现 NaN 错误！")
                    
                    user_response = input("\n是否继续训练（使用占位符数据）？这可能导致训练失败 [y/N]: ").strip().lower()
                    if user_response != 'y':
                        raise RuntimeError("训练已取消。请先运行 get_scale.py 生成 mask_scales 数据。")
                    
                    # 如果用户坚持继续，创建随机的mask_scales作为占位符（比全0好）
                    print(f"创建随机占位符 mask_scales 文件...")
                    for name in missing_scales:
                        scale_file = os.path.join(mask_scales_dir, name + '.pt')
                        # 创建一些随机的scale值（模拟真实数据分布）
                        # 通常mask scales在0.01到1.0之间
                        num_scales = torch.randint(5, 20, (1,)).item()  # 随机5-20个scales
                        random_scales = torch.rand(num_scales) * 0.99 + 0.01  # 0.01-1.0
                        torch.save(random_scales, scale_file)
                    print(f"已创建 {len(missing_scales)} 个占位符文件（警告：这些是随机数据！）")
            
            # 构建参数
            temp_parser1 = ArgumentParser()
            model_args = ModelParams(parser=temp_parser1, sentinel=True)
            model_args.model_path = self.model_path
            model_args.source_path = self.source_path
            model_args.images = "images"
            model_args.white_background = False
            model_args.sh_degree = 3
            model_args.feature_dim = 32
            model_args.eval = False
            model_args.need_features = False
            model_args.need_masks = True
            model_args.data_device = "cuda"
            model_args.resolution = -1
            if hasattr(model_args, '_resolution'):
                model_args._resolution = -1
            
            temp_parser2 = ArgumentParser()
            opt_args = OptimizationParams(parser=temp_parser2)
            opt_args.iterations = iterations
            opt_args.feature_lr = feature_lr
            opt_args.num_sampled_rays = num_sampled_rays
            opt_args.smooth_K = smooth_K
            opt_args.scale_aware_dim = scale_aware_dim
            opt_args.rfn = rfn
            opt_args.ray_sample_rate = 0
            
            temp_parser3 = ArgumentParser()
            pipe_args = PipelineParams(parser=temp_parser3)
            pipe_args.convert_SHs_python = False
            pipe_args.compute_cov3D_python = False
            pipe_args.debug = False
            
            # 清理GPU内存，为训练做准备
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 执行训练
            training(
                model_args,
                opt_args,
                pipe_args,
                iteration=-1,
                saving_iterations=[],
                checkpoint_iterations=[],
                debug_from=-1,
                log_callback=log_callback,
                progress_callback=callback
            )
            
            return True
        except Exception as e:
            print(f"训练对比特征失败: {e}")
            traceback.print_exc()
            return False
    
    def extract_sam_masks(self, sam_checkpoint_path: str, downsample: int = 1, 
                         sam_arch: str = "vit_h", downsample_type: str = "image",
                         max_long_side: int = 1024, callback=None):
        """
        提取SAM segment everything masks
        
        Args:
            sam_checkpoint_path: SAM模型路径
            downsample: 下采样比例
            sam_arch: SAM架构 ('vit_h', 'vit_l', 'vit_b')
            downsample_type: 下采样类型 ('image' 或 'mask')
            callback: 进度回调函数 callback(current, total)
        
        Returns:
            bool: 是否成功
        """
        if not self.source_path:
            print("错误: 需要先设置source_path")
            return False
        
        try:
            sys.path.insert(0, SAGA_ROOT)
            
            # 导入必要的模块
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            from PIL import Image
            import cv2
            from tqdm import tqdm
            
            print(f"\n初始化SAM模型 ({sam_arch})...")
            if callback:
                callback(0, 100, "初始化SAM模型...")
            
            # 在加载SAM之前，清理GPU显存
            print("清理GPU显存以加载SAM模型...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # 如果有Gaussian模型在GPU上，先移到CPU
            if hasattr(self, 'gaussians') and self.gaussians is not None:
                try:
                    # 临时移到CPU释放显存
                    print("暂时释放Gaussian模型显存...")
                    # 不完全卸载，只是清理缓存
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"清理Gaussian显存时出错: {e}")
            
            sam = sam_model_registry[sam_arch](checkpoint=sam_checkpoint_path).to('cuda')
            predictor = SamPredictor(sam)
            
            # 使用更保守的参数以减少内存占用
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=24,  # 从32降到24，减少采样点
                pred_iou_thresh=0.88,
                box_nms_thresh=0.7,
                stability_score_thresh=0.95,
                crop_n_layers=0,  # 不使用crop layer以节省显存
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            
            downsample_manually = False
            if downsample == 1 or downsample_type == 'mask':
                image_dir = os.path.join(self.source_path, 'images')
            else:
                image_dir = os.path.join(self.source_path, 'images_' + str(downsample))
                if not os.path.exists(image_dir):
                    image_dir = os.path.join(self.source_path, 'images')
                    downsample_manually = True
                    print("未找到下采样图像目录，将手动下采样")
            
            if not os.path.exists(image_dir):
                print(f"错误: 找不到图像目录: {image_dir}")
                return False
            
            output_dir = os.path.join(self.source_path, 'sam_masks')
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"开始提取SAM masks...")
            print(f"图像目录: {image_dir}")
            print(f"输出目录: {output_dir}")
            
            image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))])
            total = len(image_files)
            
            # 内存优化：设置最大长边限制以减少内存使用
            # 对于6GB显存的GPU，使用更保守的图像大小
            max_long_side = 800  # 降低到800像素以节省显存
            
            for idx, path in enumerate(image_files):
                if callback:
                    callback(idx, total, f"处理图像 {idx+1}/{total}: {path}")
                
                name = os.path.splitext(path)[0]
                img = cv2.imread(os.path.join(image_dir, path))
                
                if img is None:
                    print(f"警告: 无法读取图像 {path}，跳过")
                    continue
                
                # 预先下采样以减少内存使用
                if downsample_manually:
                    img = cv2.resize(img, 
                                    dsize=(img.shape[1] // downsample, img.shape[0] // downsample),
                                    interpolation=cv2.INTER_LINEAR)
                
                # 进一步限制图像大小以防止OOM
                h, w = img.shape[:2]
                long_side = max(h, w)
                if long_side > max_long_side:
                    scale = max_long_side / long_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"图像 {path} 从 {w}x{h} 调整为 {new_w}x{new_h} 以节省内存")
                
                # 使用no_grad减少内存
                with torch.no_grad():
                    retry_count = 0
                    max_retries = 3
                    masks = None
                    current_img = img.copy()
                    
                    while retry_count < max_retries:
                        try:
                            # 清理GPU缓存
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            
                            masks = mask_generator.generate(current_img)
                            break  # 成功，退出循环
                            
                        except torch.cuda.OutOfMemoryError as e:
                            retry_count += 1
                            print(f"GPU内存不足 (尝试 {retry_count}/{max_retries})，清理缓存并缩小图像...")
                            
                            # 强制清理
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            if retry_count >= max_retries:
                                print(f"图像 {path} 在{max_retries}次尝试后仍然失败，跳过此图像")
                                break
                            
                            # 逐步缩小图像
                            h, w = current_img.shape[:2]
                            scale_factor = 0.7  # 每次缩小到70%
                            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                            
                            if new_w < 256 or new_h < 256:
                                print(f"图像已缩小到最小尺寸，无法继续，跳过此图像")
                                break
                            
                            current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            print(f"  → 缩小图像到 {new_w}x{new_h}")
                            
                            # 等待一下让GPU完全释放
                            import time
                            time.sleep(0.5)
                    
                    if masks is None:
                        print(f"跳过图像 {path}")
                        continue
                
                mask_list = []
                
                for m in masks:
                    # 在CPU上处理以减少GPU内存
                    m_seg = m['segmentation']
                    m_score = torch.from_numpy(m_seg).float()
                    
                    if downsample_type == 'mask' and downsample > 1:
                        m_score = torch.nn.functional.interpolate(
                            m_score.unsqueeze(0).unsqueeze(0),
                            size=(img.shape[0] // downsample, img.shape[1] // downsample),
                            mode='bilinear', align_corners=False).squeeze()
                        m_score[m_score >= 0.5] = 1
                        m_score[m_score != 1] = 0
                    
                    m_score = m_score.bool()
                    
                    if len(m_score.unique()) < 2:
                        continue
                    else:
                        mask_list.append(m_score.cpu())  # 保存到CPU
                
                if mask_list:
                    masks_tensor = torch.stack(mask_list, dim=0)
                    torch.save(masks_tensor, os.path.join(output_dir, name + '.pt'))
                    del masks_tensor, mask_list, masks
                
                # 定期清理GPU内存
                if (idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            if callback:
                callback(total, total, "SAM masks提取完成")
            
            print(f"\nSAM masks提取完成，保存在: {output_dir}")
            return True
            
        except Exception as e:
            print(f"提取SAM masks失败: {e}")
            traceback.print_exc()
            return False
    
    def get_mask_scales(self, callback=None):
        """
        获取mask scales（需要先有sam_masks和3DGS模型）
        
        Args:
            callback: 进度回调函数 callback(current, total, message)
        
        Returns:
            bool: 是否成功
        """
        if not self.source_path:
            print("错误: 需要先设置source_path")
            return False
        
        if not self.model_path:
            print("错误: 需要先加载模型")
            return False
        
        try:
            sys.path.insert(0, SAGA_ROOT)
            
            # 检查sam_masks是否存在
            sam_masks_dir = os.path.join(self.source_path, 'sam_masks')
            if not os.path.exists(sam_masks_dir):
                print(f"错误: 找不到sam_masks目录: {sam_masks_dir}")
                print("请先运行extract_sam_masks提取SAM masks")
                return False
            
            from arguments import ModelParams, PipelineParams
            from argparse import ArgumentParser
            import cv2
            from tqdm import tqdm
            import gaussian_renderer
            
            # 检查render_with_depth是否可用
            try:
                # 尝试导入depth模块来检查是否可用
                from gaussian_renderer import render_with_depth
                # 尝试创建一个测试调用来确认模块真的可用
            except (ImportError, AttributeError) as e:
                print(f"\n错误: 缺少必需的模块 diff_gaussian_rasterization_depth")
                print("计算mask scales需要此模块。")
                print("\n检测到已编译的模块可能是为其他Python版本编译的，需要重新编译。")
                print("\n请按照以下步骤安装：")
                depth_module_path = os.path.join(SAGA_ROOT, "submodules", "diff-gaussian-rasterization-depth")
                print(f"1. 进入模块目录:")
                print(f"   cd {depth_module_path}")
                print("2. 重新编译模块（使用当前Python环境）:")
                print("   pip install -e .")
                print("\n注意: 如果编译失败，请确保:")
                print("  - 已安装CUDA工具包")
                print("  - 已安装PyTorch（支持CUDA）")
                print("  - 环境变量正确设置")
                print("\n或者参考 SegAnyGAussians 的 README.md 了解详细安装步骤。")
                print("\n提示: 如果scale_aware_dim=-1，可以跳过mask scales计算，")
                print("训练时会自动创建空的mask_scales文件。")
                return False
            
            print("\n开始获取mask scales...")
            
            # 构建参数
            temp_parser1 = ArgumentParser()
            model_args = ModelParams(parser=temp_parser1, sentinel=True)
            model_args.model_path = self.model_path
            model_args.source_path = self.source_path
            model_args.images = "images"
            model_args.white_background = False
            model_args.sh_degree = 3
            model_args.feature_dim = 32
            model_args.eval = True
            model_args.need_features = False
            model_args.need_masks = False
            model_args.data_device = "cuda"
            model_args.resolution = -1
            if hasattr(model_args, '_resolution'):
                model_args._resolution = -1
            
            temp_parser2 = ArgumentParser()
            pipe_args = PipelineParams(parser=temp_parser2)
            
            # 加载场景
            scene_gaussians = GaussianModel(model_args.sh_degree)
            scene = Scene(model_args, scene_gaussians, None, load_iteration=-1, 
                         shuffle=False, mode='eval', target='scene')
            
            # 加载图像和masks
            images_dir = os.path.join(self.source_path, 'images')
            if not os.path.exists(images_dir):
                print(f"错误: 找不到图像目录: {images_dir}")
                return False
            
            images_masks = {}
            image_files = sorted([f for f in os.listdir(images_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG'))])
            
            print("加载SAM masks...")
            for image_path in tqdm(image_files):
                name = os.path.splitext(image_path)[0]
                mask_file = os.path.join(sam_masks_dir, name + '.pt')
                if os.path.exists(mask_file):
                    masks = torch.load(mask_file)
                    images_masks[name] = masks.cpu().float()
            
            output_dir = os.path.join(self.source_path, 'mask_scales')
            os.makedirs(output_dir, exist_ok=True)
            
            cameras = scene.getTrainCameras()
            background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device='cuda')
            
            def generate_grid_index(depth):
                h, w = depth.shape
                grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
                grid = torch.stack(grid, dim=-1)
                return grid
            
            total = len(cameras)
            for it, view in enumerate(tqdm(cameras)):
                if callback:
                    callback(it, total, f"处理视图 {it+1}/{total}: {view.image_name}")
                
                # 使用no_grad减少内存
                with torch.no_grad():
                    try:
                        rendered_pkg = gaussian_renderer.render_with_depth(
                            view, scene_gaussians, pipe_args, background)
                    except ImportError as e:
                        print(f"\n错误: render_with_depth需要diff_gaussian_rasterization_depth模块")
                        print("请安装该模块或跳过mask scales计算")
                        return False
                    except Exception as e:
                        print(f"渲染视图失败 {view.image_name}: {e}")
                        continue
                
                depth = rendered_pkg['depth'].cpu().squeeze()
                del rendered_pkg  # 立即释放
                
                if view.image_name not in images_masks:
                    continue
                
                corresponding_masks = images_masks[view.image_name]
                
                grid_index = generate_grid_index(depth)
                
                points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3).cpu()
                points_in_3D[:,:,-1] = depth
                
                # 计算相机内参
                cx = depth.shape[1] / 2
                cy = depth.shape[0] / 2
                fx = cx / np.tan(cameras[0].FoVx / 2)
                fy = cy / np.tan(cameras[0].FoVy / 2)
                
                points_in_3D[:,:,0] = (grid_index[:,:,0] - cx) * depth / fx
                points_in_3D[:,:,1] = (grid_index[:,:,1] - cy) * depth / fy
                
                # 将masks移到CPU处理以减少GPU内存
                corresponding_masks_cpu = corresponding_masks.cpu()
                upsampled_mask = torch.nn.functional.interpolate(
                    corresponding_masks_cpu.unsqueeze(1), mode='bilinear',
                    size=(depth.shape[0], depth.shape[1]), align_corners=False)
                
                eroded_masks = torch.conv2d(
                    upsampled_mask.float(),
                    torch.full((3, 3), 1.0).view(1, 1, 3, 3),
                    padding=1,
                )
                eroded_masks = (eroded_masks >= 5).squeeze()
                
                scale = torch.zeros(len(corresponding_masks))
                for mask_id in range(len(corresponding_masks)):
                    point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
                    if len(point_in_3D_in_mask) > 0:
                        scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()
                
                torch.save(scale, os.path.join(output_dir, view.image_name + '.pt'))
                del depth, grid_index, points_in_3D, upsampled_mask, eroded_masks, scale
                
                # 定期清理GPU内存
                if (it + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            if callback:
                callback(total, total, "Mask scales计算完成")
            
            print(f"\nMask scales计算完成，保存在: {output_dir}")
            return True
            
        except Exception as e:
            print(f"获取mask scales失败: {e}")
            traceback.print_exc()
            return False
    
    def prepare_all_data(self, sam_checkpoint_path: str, downsample: int = 1,
                        sam_arch: str = "vit_h", downsample_type: str = "image",
                        max_long_side: int = 1024, callback=None):
        """
        一键准备所有数据（SAM masks + mask scales）
        
        Args:
            sam_checkpoint_path: SAM模型路径
            downsample: 下采样比例
            sam_arch: SAM架构
            downsample_type: 下采样类型
            callback: 进度回调函数 callback(current, total, message)
        
        Returns:
            bool: 是否成功
        """
        print("\n开始一键准备所有数据...")
        print("步骤1/2: 提取SAM masks...")
        
        def callback_wrapper(current, total, message=""):
            if callback:
                # 步骤1占50%
                callback(current, total * 2, f"[步骤1] {message}")
        
        if not self.extract_sam_masks(sam_checkpoint_path, downsample, sam_arch,
                                     downsample_type, max_long_side, callback_wrapper):
            return False
        
        print("\n步骤2/2: 计算mask scales...")
        
        def callback_wrapper2(current, total, message=""):
            if callback:
                # 步骤2占50%，从50%开始
                callback(current + total, total * 2, f"[步骤2] {message}")
        
        if not self.get_mask_scales(callback_wrapper2):
            return False
        
        print("\n所有数据准备完成！")
        return True
    
    def add_point_prompt(self, x: int, y: int, label: int = 1):
        """
        添加点提示
        
        Args:
            x: 图像x坐标
            y: 图像y坐标
            label: 标签 (1=正点, 0=负点)
        """
        self.point_prompts.append((x, y, label))
    
    def clear_point_prompts(self):
        """清空点提示"""
        self.point_prompts = []
    
    def segment_from_points(self, view_idx: int, score_thresh: float = 0.5, 
                           scale: float = 1.0) -> torch.Tensor:
        """
        基于点提示进行3D分割
        
        Args:
            view_idx: 视图索引
            score_thresh: 相似度阈值
            scale: 3D尺度
            
        Returns:
            分割掩码 (torch.Tensor, bool)
        """
        if render_contrastive_feature is None:
            raise ImportError("render_contrastive_feature未导入")
        if not self.feature_gaussians:
            raise ValueError("请先加载特征模型")
        
        if not self.point_prompts:
            raise ValueError("请先添加点提示")
        
        try:
            # 获取当前视图
            view = self.cameras[view_idx]
            
            # 渲染特征
            with torch.no_grad():
                render_pkg = render_contrastive_feature(
                    view,
                    self.feature_gaussians,
                    self.pipeline,
                    torch.zeros([32], dtype=torch.float32, device="cuda"),
                    norm_point_features=True,
                    smooth_type='traditional',
                    smooth_K=16
                )
                rendered_features = render_pkg["render"]  # [C, H, W]
            
            # 获取点提示对应的特征
            h, w = rendered_features.shape[1], rendered_features.shape[2]
            prompt_features = []
            
            for x, y, label in self.point_prompts:
                # 转换为特征图坐标
                fx = int(x * w / view.image_width) if view.image_width > 0 else x
                fy = int(y * h / view.image_height) if view.image_height > 0 else y
                fx = max(0, min(fx, w - 1))
                fy = max(0, min(fy, h - 1))
                
                feat = rendered_features[:, fy, fx]  # [C]
                if label == 1:
                    prompt_features.append(feat)
            
            if not prompt_features:
                raise ValueError("没有有效的正点提示")
            
            # 计算平均特征
            avg_feature = torch.stack(prompt_features).mean(dim=0)  # [C]
            avg_feature = avg_feature / (avg_feature.norm() + 1e-9)
            
            # 计算所有高斯的相似度
            point_features = self.feature_gaussians.get_smoothed_point_features(K=16, dropout=0.5)
            point_features = torch.nn.functional.normalize(point_features, dim=-1, p=2)
            
            # 计算相似度
            similarities = (point_features @ avg_feature.unsqueeze(-1)).squeeze(-1)
            
            # 应用阈值和尺度
            mask = similarities > score_thresh
            
            # 保存当前掩码和历史
            if self.current_mask is not None:
                self.segmentation_history.append(self.current_mask.clone())
            self.current_mask = mask
            
            return mask
        except Exception as e:
            print(f"分割失败: {e}")
            traceback.print_exc()
            raise
    
    def render_similarity_map(self, view_idx: int, score_thresh: float = 0.5, 
                             scale: float = 1.0) -> np.ndarray:
        """
        渲染相似度图
        
        Args:
            view_idx: 视图索引
            score_thresh: 相似度阈值
            scale: 3D尺度
            
        Returns:
            相似度图 (H, W, 3) numpy array
        """
        if render_contrastive_feature is None or cv2 is None:
            return None
        if not self.feature_gaussians or not self.point_prompts:
            return None
        
        try:
            view = self.cameras[view_idx]
            
            # 渲染特征
            with torch.no_grad():
                render_pkg = render_contrastive_feature(
                    view,
                    self.feature_gaussians,
                    self.pipeline,
                    torch.zeros([32], dtype=torch.float32, device="cuda"),
                    norm_point_features=True,
                    smooth_type='traditional',
                    smooth_K=16
                )
                rendered_features = render_pkg["render"]  # [C, H, W]
            
            # 获取点提示特征
            h, w = rendered_features.shape[1], rendered_features.shape[2]
            prompt_features = []
            
            for x, y, label in self.point_prompts:
                if label == 1:
                    fx = int(x * w / view.image_width) if view.image_width > 0 else x
                    fy = int(y * h / view.image_height) if view.image_height > 0 else y
                    fx = max(0, min(fx, w - 1))
                    fy = max(0, min(fy, h - 1))
                    feat = rendered_features[:, fy, fx]
                    prompt_features.append(feat)
            
            if not prompt_features:
                return None
            
            avg_feature = torch.stack(prompt_features).mean(dim=0)
            avg_feature = avg_feature / (avg_feature.norm() + 1e-9)
            
            # 计算相似度
            similarities = (rendered_features.transpose(0, 1).transpose(1, 2) @ avg_feature.unsqueeze(-1)).squeeze(-1)
            similarities = similarities.cpu().numpy()
            
            # 处理NaN和无穷大值
            if not np.isfinite(similarities).all():
                print(f"警告: 相似度数据包含非有限值，将替换为0")
                similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 归一化到0-1
            similarities = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-9)
            
            # 再次检查归一化后的值
            if not np.isfinite(similarities).all():
                similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 转换为彩色图
            similarity_map = cv2.applyColorMap((similarities * 255).astype(np.uint8), cv2.COLORMAP_JET)
            similarity_map = cv2.cvtColor(similarity_map, cv2.COLOR_BGR2RGB)
            
            return similarity_map
        except Exception as e:
            print(f"渲染相似度图失败: {e}")
            traceback.print_exc()
            return None
    
    def render_pca_features(self, view_idx: int) -> np.ndarray:
        """
        渲染PCA特征可视化
        
        Args:
            view_idx: 视图索引
            
        Returns:
            PCA特征图 (H, W, 3) numpy array
        """
        if render_contrastive_feature is None or PCA is None:
            return None
        if not self.feature_gaussians:
            return None
        
        try:
            view = self.cameras[view_idx]
            
            # 渲染特征
            with torch.no_grad():
                render_pkg = render_contrastive_feature(
                    view,
                    self.feature_gaussians,
                    self.pipeline,
                    torch.zeros([32], dtype=torch.float32, device="cuda"),
                    norm_point_features=True,
                    smooth_type='traditional',
                    smooth_K=16
                )
                rendered_features = render_pkg["render"]  # [C, H, W]
            
            # 转换为numpy
            features = rendered_features.transpose(0, 1).transpose(1, 2).cpu().numpy()  # [H, W, C]
            h, w, c = features.shape
            
            # PCA降维到3维
            features_flat = features.reshape(-1, c)
            
            # 处理NaN值：将NaN替换为0（或使用均值，但对于图像用0更合适）
            if np.isnan(features_flat).any():
                print(f"警告: 特征数据包含 {np.isnan(features_flat).sum()} 个NaN值，将替换为0")
                features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查是否还有无穷大值
            if not np.isfinite(features_flat).all():
                print(f"警告: 特征数据包含非有限值，将替换为0")
                features_flat = np.nan_to_num(features_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            pca = PCA(n_components=3)
            features_3d = pca.fit_transform(features_flat)
            
            # 归一化到0-1
            for i in range(3):
                min_val, max_val = features_3d[:, i].min(), features_3d[:, i].max()
                if max_val > min_val:
                    features_3d[:, i] = (features_3d[:, i] - min_val) / (max_val - min_val)
            
            # 重塑回图像形状
            pca_image = features_3d.reshape(h, w, 3)
            
            return (pca_image * 255).astype(np.uint8)
        except Exception as e:
            print(f"渲染PCA特征失败: {e}")
            traceback.print_exc()
            return None
    
    def cluster_3d(self, min_cluster_size: int = 50, min_samples: int = 10) -> torch.Tensor:
        """
        3D聚类
        
        Args:
            min_cluster_size: 最小聚类大小
            min_samples: 最小样本数
            
        Returns:
            聚类标签 (torch.Tensor)
        """
        if HDBSCAN is None:
            raise ImportError("HDBSCAN未导入")
        if not self.feature_gaussians:
            raise ValueError("请先加载特征模型")
        
        try:
            # 获取特征
            features = self.feature_gaussians.get_smoothed_point_features(K=16, dropout=0.5)
            features_np = features.detach().cpu().numpy()
            
            # HDBSCAN聚类
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clusterer.fit_predict(features_np)
            
            self.cluster_labels = torch.from_numpy(labels).cuda()
            
            return self.cluster_labels
        except Exception as e:
            print(f"3D聚类失败: {e}")
            traceback.print_exc()
            raise
    
    def render_cluster(self, view_idx: int) -> np.ndarray:
        """
        渲染聚类结果
        
        Args:
            view_idx: 视图索引
            
        Returns:
            聚类可视化图像 (H, W, 3) numpy array
        """
        if render is None:
            raise ImportError("render未导入")
        if self.cluster_labels is None:
            raise ValueError("请先执行3D聚类")
        
        try:
            view = self.cameras[view_idx]
            
            # 准备颜色
            unique_labels = torch.unique(self.cluster_labels)
            colors = torch.zeros((len(self.cluster_labels), 3), device="cuda")
            
            # 为每个聚类分配随机颜色
            import random
            for i, label in enumerate(unique_labels):
                if label >= 0:  # 有效聚类
                    color = [random.random(), random.random(), random.random()]
                    colors[self.cluster_labels == label] = torch.tensor(color, device="cuda")
            
            # 渲染
            with torch.no_grad():
                render_pkg = render(
                    view,
                    self.gaussians,
                    self.pipeline,
                    self.background,
                    override_color=colors
                )
                cluster_image = render_pkg["render"]
            
            cluster_image = cluster_image.permute(1, 2, 0).detach().cpu().numpy()
            cluster_image = np.clip(cluster_image, 0, 1)
            
            return (cluster_image * 255).astype(np.uint8)
        except Exception as e:
            print(f"渲染聚类失败: {e}")
            traceback.print_exc()
            return None
    
    def render_segmentation(self, view_idx: int) -> np.ndarray:
        """
        渲染分割结果
        
        Args:
            view_idx: 视图索引
            
        Returns:
            分割结果图像 (H, W, 3) numpy array
        """
        if render_mask is None:
            return None
        if self.current_mask is None:
            return None
        
        try:
            view = self.cameras[view_idx]
            
            with torch.no_grad():
                render_pkg = render_mask(
                    view,
                    self.gaussians,
                    self.pipeline,
                    self.background,
                    precomputed_mask=self.current_mask
                )
                seg_image = render_pkg["mask"]
            
            seg_image = seg_image.permute(1, 2, 0).detach().cpu().numpy()
            seg_image = np.clip(seg_image, 0, 1)
            
            return (seg_image * 255).astype(np.uint8)
        except Exception as e:
            print(f"渲染分割结果失败: {e}")
            traceback.print_exc()
            return None
    
    def save_segmentation(self, save_path: str):
        """
        保存分割掩码
        
        Args:
            save_path: 保存路径 (.pt 或 .npy)
        """
        if self.current_mask is None:
            raise ValueError("没有分割掩码可保存")
        
        if save_path.endswith('.pt'):
            torch.save(self.current_mask, save_path)
        elif save_path.endswith('.npy'):
            np.save(save_path, self.current_mask.cpu().numpy())
        else:
            raise ValueError("保存路径必须是 .pt 或 .npy 格式")
    
    def load_segmentation(self, load_path: str):
        """
        加载分割掩码
        
        Args:
            load_path: 加载路径 (.pt 或 .npy)
        """
        if load_path.endswith('.pt'):
            self.current_mask = torch.load(load_path)
            if isinstance(self.current_mask, torch.Tensor):
                self.current_mask = self.current_mask.cuda()
        elif load_path.endswith('.npy'):
            mask_np = np.load(load_path)
            self.current_mask = torch.from_numpy(mask_np).cuda().bool()
        else:
            raise ValueError("加载路径必须是 .pt 或 .npy 格式")
    
    def undo_segmentation(self):
        """撤销上一次分割"""
        if self.segmentation_history:
            self.current_mask = self.segmentation_history.pop()
            return True
        return False
    
    def clear_segmentation(self):
        """清空分割"""
        self.current_mask = None
        self.segmentation_history = []
        self.point_prompts = []
    
    def render_rgb(self, view_idx: int) -> np.ndarray:
        """
        渲染RGB图像
        
        Args:
            view_idx: 视图索引
            
        Returns:
            RGB图像 (H, W, 3) numpy array
        """
        if render is None:
            return None
        if not self.gaussians:
            return None
        
        try:
            view = self.cameras[view_idx]
            
            with torch.no_grad():
                render_pkg = render(view, self.gaussians, self.pipeline, self.background)
                rgb_image = render_pkg["render"]
            
            rgb_image = rgb_image.permute(1, 2, 0).detach().cpu().numpy()
            rgb_image = np.clip(rgb_image, 0, 1)
            
            return (rgb_image * 255).astype(np.uint8)
        except Exception as e:
            print(f"渲染RGB失败: {e}")
            traceback.print_exc()
            return None
