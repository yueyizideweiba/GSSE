# GSSE - Gaussian Splatting Semantic Editor

A semantic segmentation and editing tool based on 3D Gaussian Splatting, supporting interactive object segmentation and scene editing.

## ✨ Key Features

- **Interactive Segmentation**: Supports point-click and text-prompt based 3D object segmentation
- **Multi-view Voting**: Improves segmentation accuracy through multi-view consistency
- **GUI Interface**: Intuitive graphical user interface
- **Multiple SAM Models**: Supports SAM (ViT-H/L/B) and FastSAM
- **PLY Export**: Segmentation results can be exported in standard PLY format
- **GIS Support**: Supports EXIF extraction and GIS coordinate conversion
- **Cesium Integration**: Supports visualization of 3D Gaussian point clouds in Cesium
- **Format Conversion**: Supports conversion from PLY to Splat format

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd GSSE
```

### 2. Environment Setup

```bash
# Create conda environment
conda create -n gsse python=3.10
conda activate gsse

# Install PyTorch (CUDA 11.6)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for Cesium visualization)
npm install
```

### 3. Install Dependencies

This project depends on the following third-party projects, which need to be manually downloaded and configured:

#### 3.1 Install 3D Gaussian Splatting

```bash
# Clone 3D Gaussian Splatting repository
cd dependencies
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
mv gaussian-splatting ../gaussiansplatting
cd ..

# Compile extensions
cd gaussiansplatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
cd ..
```

#### 3.2 Install COLMAP

**Method 1: Using Pre-compiled Version (Recommended)**

```bash
# Download COLMAP
cd dependencies
wget https://github.com/colmap/colmap/releases/download/3.9/COLMAP-3.9-linux-cuda.tar.gz
tar -xzf COLMAP-3.9-linux-cuda.tar.gz
mv COLMAP-3.9-linux-cuda ../colmap
cd ..
```

**Method 2: Compile from Source**

```bash
# Clone COLMAP repository
cd dependencies
git clone https://github.com/colmap/colmap.git
mv colmap ..
cd ../colmap

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j8
sudo make install
cd ../..
```

#### 3.3 Install Model Dependencies

```bash
# Create dependency directories
mkdir -p dependencies/sam_ckpt
mkdir -p dependencies/GroundingDINO/weights

# Download SAM model weights
cd dependencies/sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # 2.4GB
# Or download smaller model
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  # 375MB
cd ../..

# Download GroundingDINO weights
cd dependencies/GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../..

# Install GroundingDINO
cd dependencies
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ../..

# Install SegAnyGaussians (Optional)
cd dependencies
git clone https://github.com/Pointcept/SegAnyGaussians.git SegAnyGAussians
cd SegAnyGAussians
pip install -e .
cd ../..
```

#### 3.4 Install Cesium Libraries (Optional)

```bash
# Download Cesium library
mkdir -p lib
cd lib
wget https://github.com/CesiumGS/cesium/releases/download/1.110/Cesium-1.110.zip
unzip Cesium-1.110.zip
mv Cesium-1.110 Cesium
cd ..

# Install gaussian-splats-3d (if using)
cd dependencies
git clone https://github.com/mkkellogg/gaussian-splats-3d.git cesium-gaussian-splatting
cd cesium-gaussian-splatting
npm install
npm run build
cd ../..
```

### 4. Verify Installation

```bash
# Check Python dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gaussiansplatting; print('3DGS installed')"

# Check COLMAP
colmap -h

# Check SAM models
ls dependencies/sam_ckpt/*.pth
```

### 5. Prepare Training Data

Place your image data in the `input/` directory:

```bash
input/
└── your_scene/
    └── images/
        ├── image_001.jpg
        ├── image_002.jpg
        └── ...
```

## Usage

### 1. Train 3D Gaussian Splatting Model

```bash
python train.py -s <your-data-path> -m <output-path>
```

### 2. Launch GUI for Segmentation

```bash
python gsse_gui.py
```

### 3. Command Line Segmentation (Optional)

```bash
# Point-click segmentation
python run_gsse.py \
  --model_path <training-output-path> \
  --iteration 7000 \
  --prompt_type point \
  --point 300,180

# Text prompt segmentation
python run_gsse.py \
  --model_path <training-output-path> \
  --iteration 7000 \
  --prompt_type text \
  --text "a chair"
```

## 📁 Project Structure

This project only contains core code. Dependencies need to be manually configured according to the installation steps above.

```
GSSE/
├── gsse_gui.py                  # GUI main program (core code)
├── run_gsse.py                  # Command line segmentation tool (core code)
├── gaussian_editor.py           # Gaussian editor (core code)
├── gaussian_multi_mode_renderer.py  # Multi-mode renderer (core code)
├── seg_utils.py                 # Segmentation utility functions (core code)
├── saga_module.py               # SegAnyGaussians module (core code)
├── sog.py                       # Self-Organizing Gaussians (core code)
├── train.py                     # Training script (core code)
├── colmap.py                    # COLMAP interface (core code)
├── exif_geo_extractor.py        # EXIF geographic information extraction (core code)
├── gis_converter.py             # GIS coordinate conversion (core code)
├── ply_to_splat_converter.py    # PLY to Splat format conversion (core code)
├── cesium_widget.py             # Cesium visualization component (core code)
├── cesium_model_manager.py      # Cesium model manager (core code)
├── cesium_viewer.html           # Cesium viewer HTML (core code)
├── requirements.txt             # Python dependency list
├── package.json                 # Node.js dependency list
├── MODEL_SETUP.md              # Model setup guide
│
├── gaussiansplatting/           # ⚠️ Requires manual installation (see installation step 3.1)
├── dependencies/                # ⚠️ Requires manual installation (see installation step 3.3)
│   ├── sam_ckpt/               # SAM model weights
│   ├── GroundingDINO/          # GroundingDINO
│   └── SegAnyGAussians/        # SegAnyGaussians
├── colmap/                      # ⚠️ Requires manual installation (see installation step 3.2)
└── lib/                         # ⚠️ Requires manual installation (see installation step 3.4)
    ├── Cesium/                 # Cesium library
    └── gaussian-splats-3d/     # 3D Gaussian point cloud library
```

## ⚙️ Parameter Description

- `--sam_model`: SAM model type (vit_h/vit_l/vit_b/fastsam_s/fastsam_x)
- `--threshold`: Multi-view voting threshold (default 0.7)
- `--gd_interval`: Gaussian decomposition interval (-1 to disable)
- `--mask_id`: SAM candidate mask ID (0/1/2)

## 🔧 Advanced Features

### Cesium Visualization

The project integrates Cesium for visualizing 3D Gaussian point clouds in a web browser:

```bash
# Use Cesium viewer
python cesium_widget.py
```

### Format Conversion

Supports conversion from PLY format to Splat format:

```bash
python ply_to_splat_converter.py <ply-file> <output-directory>
```

## 📝 Notes

- **GPU Requirements**: At least 8GB VRAM, 16GB or more recommended
- **Environment Configuration**: It is recommended to strictly follow requirements.txt for environment setup to avoid dependency conflicts
- **Dependency Installation**: This project only contains core code. All dependencies (colmap, gaussiansplatting, dependencies, etc.) need to be manually configured according to the installation steps
- **Model Weights**: Model weight files are large (375MB-2.4GB) and need to be downloaded in advance
- **First Run**: Before the first run, please ensure all dependencies are correctly installed and configured
