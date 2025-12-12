# Initialize conda
# Try to find conda installation
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -n "$CONDA_PREFIX" ]; then
    # If conda is already in PATH, try to source from CONDA_PREFIX
    if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    fi
else
    # Try to find conda using which
    CONDA_BASE=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    else
        echo "警告: 无法找到 conda，尝试直接运行..."
    fi
fi

# Activate sam3 conda environment
echo "正在激活 conda 环境: sam3"
conda activate sam3 2>/dev/null || {
    echo "错误: 无法激活 conda 环境 'sam3'"
    echo "请确保已创建该环境: conda create -n sam3"
    exit 1
}

# Ensure we are in the script's directory or set PYTHONPATH correctly
cd "$(dirname "$0")"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Checking dependencies..."
python -c "import flask" 2>/dev/null || pip install flask
python -c "import flask_cors" 2>/dev/null || pip install flask-cors
python -c "import cv2" 2>/dev/null || pip install opencv-python

echo "Starting SAM3 API Server..."
# Run from parent directory context if needed for module resolution, or adjust python path
# sam3 package expects to be imported. If we are in sam3/sam3, that's inside the package.
# The user's folder structure is:
# sam3/
#   sam3/ (package source)
#   main.py
#   ...
#   api_server.py (I placed it in sam3 root)
#
# So if I placed api_server.py in sam3/, running `python api_server.py` there should work 
# assuming 'sam3' package is installed in editable mode (pip install -e .) or in pythonpath.
# The user said "file in sam3 folder", so I put it in the root `sam3/` folder (alongside `main.py`).

python api_server.py "$@"