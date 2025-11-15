#!/usr/bin/env bash
source ~/.bashrc 

set -euo pipefail

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 切换到项目目录（使用相对路径）
cd "$SCRIPT_DIR"

# 清理可能存在的OpenCV Qt插件路径环境变量
# 这些变量会在Python脚本中被正确处理
unset QT_PLUGIN_PATH 2>/dev/null || true
unset QT_QPA_PLATFORM_PLUGIN_PATH 2>/dev/null || true

# 使用 conda run 避免在脚本内显式 source 激活脚本
# 注意：Qt插件路径会在Python脚本内部正确设置
exec conda run -n SAGS python "$SCRIPT_DIR/gsse_gui.py"


