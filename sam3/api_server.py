import os
import io
import base64
import json
import logging
import time
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Global model variables
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
current_device = device  # 当前实际使用的设备

def switch_to_cpu():
    """将模型切换到CPU设备"""
    global model, processor, current_device
    if current_device == "cpu":
        return  # 已经在CPU上了
    
    logger.warning("检测到CUDA内存不足，正在切换到CPU模式...")
    try:
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 强制清理所有CUDA缓存
            torch.cuda.ipc_collect()
        
        # 将模型移动到CPU - 使用更彻底的方法
        if model is not None:
            # 先确保模型在评估模式
            model.eval()
            
            # 使用.to()方法移动模型（这会递归移动所有子模块）
            model = model.to("cpu")
            
            # 递归检查并移动所有子模块到CPU
            def move_module_to_cpu(module):
                """递归地将模块及其所有子模块移动到CPU"""
                module = module.cpu()
                for child in module.children():
                    move_module_to_cpu(child)
                return module
            
            model = move_module_to_cpu(model)
            
            # 双重检查：确保所有参数和缓冲区都在CPU上
            for name, param in model.named_parameters():
                if param.is_cuda:
                    logger.debug(f"移动参数 {name} 到CPU")
                    param.data = param.data.cpu()
                    if param.grad is not None and param.grad.is_cuda:
                        param.grad = param.grad.cpu()
            
            for name, buffer in model.named_buffers():
                if buffer.is_cuda:
                    logger.debug(f"移动缓冲区 {name} 到CPU")
                    buffer.data = buffer.data.cpu()
            
            # 验证所有参数和缓冲区都在CPU上
            cuda_params = [name for name, param in model.named_parameters() if param.is_cuda]
            cuda_buffers = [name for name, buffer in model.named_buffers() if buffer.is_cuda]
            if cuda_params or cuda_buffers:
                logger.warning(f"仍有参数/缓冲区在CUDA上: params={cuda_params[:5]}..., buffers={cuda_buffers[:5]}...")
                # 强制移动
                for name in cuda_params:
                    param = dict(model.named_parameters())[name]
                    param.data = param.data.cpu()
                for name in cuda_buffers:
                    buffer = dict(model.named_buffers())[name]
                    buffer.data = buffer.data.cpu()
            
            # 确保所有注册的缓冲区也在CPU上
            for name, buffer in model.named_buffers():
                if buffer.is_cuda:
                    setattr(model, name.replace('.', '_'), buffer.cpu())
            
            # 清除decoder的坐标缓存（这些缓存可能包含CUDA tensor）
            def clear_decoder_cache(module):
                """递归清除所有decoder模块的缓存"""
                if hasattr(module, 'coord_cache'):
                    module.coord_cache = {}
                    logger.debug("清除decoder coord_cache")
                if hasattr(module, 'compilable_cord_cache'):
                    module.compilable_cord_cache = None
                    logger.debug("清除decoder compilable_cord_cache")
                if hasattr(module, 'compilable_stored_size'):
                    module.compilable_stored_size = None
                    logger.debug("清除decoder compilable_stored_size")
                for child in module.children():
                    clear_decoder_cache(child)
            
            clear_decoder_cache(model)
        
        # 重新创建processor，使用CPU设备（这会重新初始化所有内部状态）
        processor = Sam3Processor(model, device="cpu")
        current_device = "cpu"
        
        # 强制清理所有CUDA缓存和内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            # 尝试释放所有未使用的内存
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info("已成功切换到CPU模式，后续推理将在CPU上执行")
    except Exception as e:
        logger.error(f"切换到CPU模式失败: {e}", exc_info=True)
        raise e

def load_model():
    global model, processor, current_device
    logger.info(f"Loading SAM3 model on {device}...")
    try:
        model = build_sam3_image_model()
        # Ensure model is on the correct device if build_sam3_image_model doesn't handle it automatically
        # The processor also takes a device argument
        processor = Sam3Processor(model, device=device)
        current_device = device
        logger.info("SAM3 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}")
        raise e

def mask_to_geojson(mask_np, label="object"):
    """
    Convert a boolean numpy mask to a GeoJSON FeatureCollection.
    The coordinates are in image pixel space (0,0 is top-left).
    """
    # Ensure mask is 2D and uint8
    mask_uint8 = mask_np.astype(np.uint8)
    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8.squeeze()
    
    # Ensure contiguous array for OpenCV
    mask_uint8 = np.ascontiguousarray(mask_uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue
            
        # contour is (N, 1, 2) -> (N, 2)
        # Flip y? usually GeoJSON/Cesium might expect different coordinate systems,
        # but the request implies "2D vector overlay" which usually maps pixel coords to 3D later.
        # We will return pixel coordinates here.
        coords = contour.squeeze().tolist()
        # Close the loop
        coords.append(coords[0])
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "id": i,
                "label": label
            }
        }
        features.append(feature)
        
    return {
        "type": "FeatureCollection",
        "features": features
    }

def process_segmentation(image, prompt, threshold=0.5, mask_threshold=0.5):
    if processor is None:
        return {"success": False, "summary": "Model not loaded"}, 500
    
    # 尝试在CUDA上运行，如果OOM则回退到CPU
    try:
        return _process_segmentation_internal(image, prompt, threshold, mask_threshold)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        error_msg = str(e)
        # 检查是否是CUDA OOM错误
        if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
            logger.warning(f"CUDA内存不足: {error_msg}")
            logger.info("尝试清理CUDA缓存并切换到CPU模式...")
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 如果当前不在CPU上，切换到CPU
            if current_device != "cpu":
                try:
                    switch_to_cpu()
                except Exception as switch_error:
                    logger.error(f"切换到CPU失败: {switch_error}")
                    return {
                        "success": False, 
                        "summary": f"CUDA内存不足且无法切换到CPU: {switch_error}"
                    }, 500
            
            # 在CPU上重试
            try:
                logger.info("在CPU模式下重新执行分割...")
                # 确保所有CUDA缓存都被清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # 强制释放所有CUDA内存
                    torch.cuda.ipc_collect()
                    # 再次清理
                    torch.cuda.empty_cache()
                
                # 重新执行分割（此时processor已经在CPU上）
                # 注意：此时image可能还是PIL Image，所以应该没问题
                # 使用force_cpu=True确保完全在CPU上执行
                return _process_segmentation_internal(image, prompt, threshold, mask_threshold, force_cpu=True)
            except Exception as cpu_error:
                logger.error(f"CPU模式下的分割也失败: {cpu_error}", exc_info=True)
                return {
                    "success": False, 
                    "summary": f"分割失败（已切换到CPU）: {cpu_error}"
                }, 500
        else:
            # 其他类型的错误
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            return {"success": False, "summary": str(e)}, 500
    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        return {"success": False, "summary": str(e)}, 500

def _process_segmentation_internal(image, prompt, threshold=0.5, mask_threshold=0.5, force_cpu=False):
    """内部分割处理函数，不处理OOM异常
    
    Args:
        force_cpu: 如果为True，强制在CPU上执行，即使current_device是cuda
    """
    global model, processor, current_device
    
    # 确保processor和设备状态一致
    if processor is None:
        raise RuntimeError("Processor not initialized")
    
    # 如果强制使用CPU，确保processor在CPU上
    target_device = "cpu" if force_cpu else current_device
    
    # 验证processor的设备设置
    if hasattr(processor, 'device') and processor.device != target_device:
        logger.warning(f"Processor设备({processor.device})与目标设备({target_device})不匹配，重新创建processor...")
        processor = Sam3Processor(model, device=target_device)
    
    # 确保图像是PIL Image格式（如果是tensor，需要先转换）
    if isinstance(image, torch.Tensor):
        # 如果图像是tensor，需要确保它在正确的设备上
        if image.is_cuda and target_device == "cpu":
            image = image.cpu()
        elif not image.is_cuda and target_device == "cuda":
            image = image.cuda()
    
    # Pre-process image
    # set_image expects PIL Image or Tensor
    # 每次调用都重新创建inference_state，确保没有残留的CUDA数据
    # 在调用前再次清理CUDA缓存（如果使用CPU）
    if target_device == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 确保model在正确的设备上
    if target_device == "cpu" and model is not None:
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.is_cuda:
            logger.warning("检测到model仍在CUDA上，强制移动到CPU...")
            model = model.to("cpu")
            processor = Sam3Processor(model, device="cpu")
    
    # 调用set_image
    # 在CPU模式下，确保所有操作在CPU上执行
    inference_state = processor.set_image(image)
    
    # 验证backbone输出在正确的设备上（递归检查所有tensor）
    def move_to_device(obj, target_device):
        """递归地将所有tensor移动到目标设备"""
        if isinstance(obj, torch.Tensor):
            if obj.device.type != target_device:
                return obj.to(target_device, non_blocking=False)
            return obj
        elif isinstance(obj, dict):
            return {k: move_to_device(v, target_device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(item, target_device) for item in obj)
        return obj
    
    # 确保整个inference_state都在正确的设备上
    inference_state = move_to_device(inference_state, target_device)
    
    # 定义深度移动函数（在函数级别，以便后续使用）
    def deep_move_to_cpu(obj, depth=0):
        """深度递归地将所有tensor移动到CPU"""
        if depth > 10:  # 防止无限递归
            return obj
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                return obj.cpu()
            return obj
        elif isinstance(obj, dict):
            return {k: deep_move_to_cpu(v, depth+1) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(deep_move_to_cpu(item, depth+1) for item in obj)
        elif hasattr(obj, '__dict__'):
            # 对于对象，尝试移动其属性
            for attr_name in dir(obj):
                if not attr_name.startswith('_'):
                    try:
                        attr = getattr(obj, attr_name)
                        if isinstance(attr, torch.Tensor) and attr.is_cuda:
                            setattr(obj, attr_name, attr.cpu())
                    except:
                        pass
        return obj
    
    # 在CPU模式下，再次深度检查并移动所有tensor
    if target_device == "cpu":
        # 递归检查backbone_out中的所有tensor
        if "backbone_out" in inference_state:
            backbone_out = inference_state["backbone_out"]
            inference_state["backbone_out"] = deep_move_to_cpu(backbone_out)
    
    # Set confidence threshold
    if threshold is not None:
         # Sam3Processor.set_confidence_threshold expects float
         # Note: current implementation of Sam3Processor might store it in self.confidence_threshold
         # but set_image resets state? No, set_image initializes state.
         # Let's check source code logic again... 
         # Sam3Processor has set_confidence_threshold method.
         processor.set_confidence_threshold(float(threshold), state=inference_state)
    
    # 在调用set_text_prompt之前，再次确保所有tensor都在正确的设备上
    inference_state = move_to_device(inference_state, target_device)
    
    # 验证model是否在正确的设备上（如果使用CPU）
    if target_device == "cpu" and model is not None:
        # 检查model的主要参数是否在CPU上
        first_param = next(model.parameters(), None)
        if first_param is not None and first_param.is_cuda:
            logger.warning("检测到model参数仍在CUDA上，强制移动到CPU...")
            model = model.to("cpu")
            # 重新创建processor以确保一致性
            processor = Sam3Processor(model, device="cpu")
        
        # 再次清理CUDA缓存，确保没有残留
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # 在CPU模式下，确保processor的find_stage也在CPU上
    if target_device == "cpu":
        # 确保processor的device确实是"cpu"
        if hasattr(processor, 'device') and processor.device != "cpu":
            logger.warning("强制processor使用CPU设备...")
            processor = Sam3Processor(model, device="cpu")
        
        # 确保processor的find_stage中的所有tensor都在CPU上
        if hasattr(processor, 'find_stage'):
            find_stage = processor.find_stage
            if hasattr(find_stage, 'img_ids') and find_stage.img_ids.is_cuda:
                find_stage.img_ids = find_stage.img_ids.cpu()
            if hasattr(find_stage, 'text_ids') and find_stage.text_ids.is_cuda:
                find_stage.text_ids = find_stage.text_ids.cpu()
            # 移动其他可能的tensor属性
            for attr_name in dir(find_stage):
                if not attr_name.startswith('_'):
                    attr = getattr(find_stage, attr_name)
                    if isinstance(attr, torch.Tensor) and attr.is_cuda:
                        setattr(find_stage, attr_name, attr.cpu())
        
        # 再次移动inference_state到CPU（以防万一）
        inference_state = move_to_device(inference_state, "cpu")
        
        # 再次清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Run inference
    # 在CPU模式下，确保所有操作都在CPU上执行
    if target_device == "cpu":
        # 最后一次深度检查并移动所有tensor到CPU
        inference_state = deep_move_to_cpu(inference_state, depth=0)
        
        # 验证backbone_out中的所有tensor都在CPU上
        if "backbone_out" in inference_state:
            backbone_out = inference_state["backbone_out"]
            cuda_tensors = []
            def find_cuda_tensors(obj, path="", depth=0):
                if depth > 10:
                    return
                if isinstance(obj, torch.Tensor):
                    if obj.is_cuda:
                        cuda_tensors.append(f"{path}: {obj.shape} on {obj.device}")
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        find_cuda_tensors(v, f"{path}.{k}" if path else k, depth+1)
                elif isinstance(obj, (list, tuple)):
                    for i, item in enumerate(obj):
                        find_cuda_tensors(item, f"{path}[{i}]" if path else f"[{i}]", depth+1)
            
            find_cuda_tensors(backbone_out)
            if cuda_tensors:
                logger.warning(f"发现backbone_out中仍有CUDA tensor: {cuda_tensors[:5]}")
                inference_state["backbone_out"] = deep_move_to_cpu(backbone_out)
        
        # 使用no_grad上下文，并确保所有tensor在CPU上
        with torch.no_grad():
            # 再次验证并移动所有tensor到CPU
            inference_state = move_to_device(inference_state, "cpu")
            try:
                inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
            except RuntimeError as e:
                if "device" in str(e).lower() or "cuda" in str(e).lower():
                    logger.error(f"设备不匹配错误: {e}")
                    # 最后一次尝试：重新创建processor并清理所有状态
                    logger.warning("最后一次尝试：重新创建processor并清理所有CUDA状态...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # 清除decoder缓存
                    def clear_decoder_cache(module):
                        """递归清除所有decoder模块的缓存"""
                        if hasattr(module, 'coord_cache'):
                            module.coord_cache = {}
                        if hasattr(module, 'compilable_cord_cache'):
                            module.compilable_cord_cache = None
                        if hasattr(module, 'compilable_stored_size'):
                            module.compilable_stored_size = None
                        for child in module.children():
                            clear_decoder_cache(child)
                    
                    clear_decoder_cache(model)
                    
                    processor = Sam3Processor(model, device="cpu")
                    # 重新调用set_image以确保所有状态都在CPU上
                    # 确保image在CPU上（如果是tensor）
                    if isinstance(image, torch.Tensor) and image.is_cuda:
                        image = image.cpu()
                    inference_state = processor.set_image(image)
                    # 递归移动所有tensor到CPU
                    inference_state = move_to_device(inference_state, "cpu")
                    if threshold is not None:
                        processor.set_confidence_threshold(float(threshold), state=inference_state)
                    # 再次确保所有tensor在CPU上
                    inference_state = move_to_device(inference_state, "cpu")
                    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
                else:
                    raise
    else:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    # Extract results
    # masks is (N, H, W) boolean tensor
    masks = inference_state["masks"]
    scores = inference_state["scores"]
    
    if len(masks) == 0:
         return {
            "success": True,
            "geojson": {"type": "FeatureCollection", "features": []},
            "summary": "No objects found."
        }, 200

    # Combine all masks for the prompt (union) or return individual features?
    # Usually "segment the building" might return one or multiple instances.
    # We will return all instances found as separate features in one GeoJSON.
    
    combined_features = []
    
    # Iterate over detected instances
    for i in range(len(masks)):
        mask = masks[i].cpu().numpy()
        score = float(scores[i].cpu().numpy())
        
        # Apply mask threshold if it were logits, but 'masks' is already boolean (>0.5)
        # if we used masks_logits we would threshold here.
        # The Sam3Processor source says: state["masks"] = out_masks > 0.5
        
        geojson_part = mask_to_geojson(mask, label=prompt)
        for feature in geojson_part["features"]:
            feature["properties"]["score"] = score
            combined_features.append(feature)
            
    final_geojson = {
        "type": "FeatureCollection",
        "features": combined_features
    }
    
    device_info = f"（使用{target_device.upper()}）" if target_device == "cpu" else ""
    return {
        "success": True,
        "geojson": final_geojson,
        "summary": f"Found {len(masks)} instances of '{prompt}'.{device_info}"
    }, 200

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    status = "ok" if model is not None else "loading"
    return jsonify({
        "status": status, 
        "description": "SAM3 Image Segmentation API", 
        "version": "1.0.0"
    })

@app.route('/api/segment', methods=['POST'])
def segment_json():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "summary": "No JSON data provided"}), 400
        
    image_b64 = data.get('image')
    prompt = data.get('prompt')
    
    if not image_b64 or not prompt:
        return jsonify({"success": False, "summary": "Missing 'image' (base64) or 'prompt'"}), 400
        
    threshold = data.get('threshold', 0.5)
    mask_threshold = data.get('mask_threshold', 0.5)
    
    try:
        # Decode base64 image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "summary": f"Invalid image data: {str(e)}"}), 400
        
    result, status_code = process_segmentation(image, prompt, threshold, mask_threshold)
    return jsonify(result), status_code

@app.route('/api/segment/file', methods=['POST'])
def segment_file():
    if 'image' not in request.files:
        return jsonify({"success": False, "summary": "No image file provided"}), 400
        
    file = request.files['image']
    prompt = request.form.get('prompt')
    
    if not prompt:
        return jsonify({"success": False, "summary": "Missing 'prompt'"}), 400
        
    threshold = request.form.get('threshold', 0.5)
    mask_threshold = request.form.get('mask_threshold', 0.5)
    
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "summary": f"Invalid image file: {str(e)}"}), 400
        
    result, status_code = process_segmentation(image, prompt, threshold, mask_threshold)
    return jsonify(result), status_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM3 Segmentation API Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    args = parser.parse_args()
    
    # Load model before starting server
    with app.app_context():
        load_model()
    
    app.run(host=args.host, port=args.port, debug=False)