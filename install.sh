#!/usr/bin/env bash
set -e

# Simple install script applying the flash-attn workaround

python -m pip install --upgrade pip setuptools wheel

# Install torch first because flash-attn requires it for building
pip install "torch>=2.4.0" "torchvision>=0.19.0"

# Install flash-attn using the recommended no-build-isolation approach
if ! pip install flash-attn --no-build-isolation; then
  # Fallback: install from Git if the wheel build still fails
  pip install git+https://github.com/Dao-AILab/flash-attention.git
fi

# Install the rest of the dependencies
pip install -r requirements.txt

