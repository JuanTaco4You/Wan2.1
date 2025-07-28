#!/usr/bin/env bash
set -e

# Simple install script applying the flash-attn workaround

python -m pip install --upgrade pip setuptools wheel

# Install flash-attn first, using the recommended no-build-isolation approach.
if ! pip install flash-attn --no-build-isolation; then
  # Fallback: install from Git if the wheel build still fails
  pip install git+https://github.com/Dao-AILab/flash-attention.git
fi

# Install the rest of the dependencies
pip install -r requirements.txt

