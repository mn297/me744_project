#!/bin/bash

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv not installed"
    exit 1
fi

# Initialize UV project with Python 3.12
if [ ! -f "pyproject.toml" ]; then
    uv init --python 3.12
else
    echo "pyproject.toml already exists"
fi

# Ensure Python 3.12 is used
if [ ! -f ".python-version" ]; then
    echo "3.12" > .python-version
fi

# Configure PyTorch CUDA for Linux/Windows
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    if [ ! -f "pyproject.toml" ]; then
        echo "pyproject.toml does not exist"
        exit 1
    fi
    cat >> pyproject.toml << 'EOF'

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchaudio = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
EOF
fi

# Install packages
uv add torch torchvision torchaudio
uv add opencv-python ultralytics
uv add numpy pandas matplotlib seaborn scipy scikit-learn
uv add stable-baselines3 gymnasium gym gym-anytrading
uv add tqdm

# jupyter stuff
uv add --dev ipykernel
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
uv run --with jupyter jupyter lab