import torch
import matplotlib.pyplot as plt
from PIL import Image
 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
 
# 加载模型（会自动读取本地 sam3.pt）
model = build_sam3_image_model()
processor = Sam3Processor(model)
 
# 加载测试图片
image = Image.open("assets/images/test_image.jpg")
 
# 设置图像（这一步会做全图编码）
inference_state = processor.set_image(image)
 
# 文本提示分割（换成你想要的词）
inference_state = processor.set_text_prompt(state=inference_state, prompt="child")
# 或者分割鞋子：prompt="shoe"
# 或者试试：prompt="foot" / "sock" / "person" / "hat" 都好使
 
# 可视化结果（我修复了官方 plot_results 没 plt.show() 的 bug）
plot_results(image, inference_state)
plt.show()  # 加上这句才能弹出图片
