#!/bin/bash

# 从零构建语言模型 - 环境安装脚本
# 使用 .venv 创建独立的 Python 虚拟环境

set -e  # 遇到错误时停止脚本

echo "========================================="
echo "开始安装 Python 环境..."
echo "========================================="

# 1. 创建虚拟环境
if [ -d ".venv" ]; then
    echo "虚拟环境已存在，跳过创建步骤..."
else
    echo "创建 .venv 虚拟环境..."
    python3 -m venv .venv
fi

# 2. 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 3. 升级 pip
echo "升级 pip..."
pip install --upgrade pip

# 4. 安装 tiktoken 0.7.0 版本
# 注意：tiktoken 0.7.0 使用 tiktoken.get_encoding() API
echo "安装 tiktoken 0.7.0 版本..."
pip install tiktoken==0.7.0

# 5. 安装 PyTorch 及相关依赖
echo "安装 PyTorch 和相关依赖..."

# 根据系统类型选择合适的安装命令
if command -v nvidia-smi &> /dev/null; then
    echo "检测到 NVIDIA GPU，安装 PyTorch (CUDA 版本)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到 NVIDIA GPU，安装 PyTorch (CPU 版本)..."
    pip install torch torchvision torchaudio
fi

# 5. 安装常用数据处理库
echo "安装常用数据处理库..."
pip install numpy pandas matplotlib jupyter tqdm

# 6. 安装其他 LLM 相关工具
echo "安装其他 LLM 相关工具..."
pip install transformers datasets accelerate

echo "========================================="
echo "安装完成！"
echo "========================================="
echo ""
echo "激活虚拟环境命令："
echo "  source .venv/bin/activate"
echo ""
echo "验证安装："
echo "  python -c \"import torch; print(f'PyTorch version: {torch.__version__}')\""
echo ""
