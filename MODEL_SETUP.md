# 模型和数据设置指南

由于GitHub有文件大小限制，以下大型文件需要单独下载配置。

## 1. SAM模型权重

下载到 `dependencies/sam_ckpt/` 目录：

```bash
cd dependencies/sam_ckpt

# SAM ViT-H (最佳质量，2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# SAM ViT-L (平衡，1.2GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# SAM ViT-B (最快，375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# FastSAM (可选，更快)
# 从 https://github.com/CASIA-IVA-Lab/FastSAM/releases 下载
```

## 2. GroundingDINO权重

下载到 `dependencies/GroundingDINO/weights/` 目录：

```bash
cd dependencies/GroundingDINO/weights

# GroundingDINO权重
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

## 3. 目录结构

设置完成后应该是这样：

```
GSSE/
├── dependencies/
│   ├── sam_ckpt/
│   │   ├── sam_vit_h_4b8939.pth
│   │   ├── sam_vit_l_0b3195.pth
│   │   └── sam_vit_b_01ec64.pth
│   └── GroundingDINO/
│       └── weights/
│           └── groundingdino_swint_ogc.pth
└── ...
```

## 4. 准备训练数据

把你的图像数据放到 `input/` 或 `gaussiansplatting/input/` 目录：

```bash
input/
└── your_scene/
    └── images/
        ├── image_001.jpg
        ├── image_002.jpg
        └── ...
```

## 5. 编译COLMAP（可选）

如果需要3D重建功能：

```bash
cd colmap
mkdir build && cd build
cmake ..
make -j8
```

## 注意事项

- SAM模型权重文件很大（375MB-2.4GB），下载需要时间
- 建议至少下载一个SAM模型（推荐vit_h或vit_b）
- 训练数据不要上传到GitHub，太大了
- 这些大文件已经在 `.gitignore` 中排除了

## 6. 验证安装

安装完成后，可以通过以下命令验证：

```bash
# 检查SAM模型文件
ls -lh dependencies/sam_ckpt/*.pth

# 检查GroundingDINO权重
ls -lh dependencies/GroundingDINO/weights/*.pth

# 检查COLMAP（如果编译了）
colmap -h
```

## 7. 常见问题

**Q: 下载速度慢怎么办？**  
A: 可以使用镜像源或使用代理。SAM模型也可以从其他镜像站下载。

**Q: 必须下载所有模型吗？**  
A: 不需要。至少下载一个SAM模型即可（推荐vit_b或vit_h）。GroundingDINO只在需要文本提示分割时使用。

**Q: COLMAP必须编译吗？**  
A: 如果只需要分割功能，可以不编译COLMAP。如果需要3D重建，建议使用预编译版本。
