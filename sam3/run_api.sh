#!/bin/bash

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