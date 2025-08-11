#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Starting build process..."

# Upgrade pip to avoid version conflicts
python -m pip install --upgrade pip

# Install wheel and setuptools first
pip install wheel setuptools

# Install requirements with specific flags to avoid build issues
pip install --no-cache-dir -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p faiss_index

# Set proper permissions
chmod -R 755 uploads || true
chmod -R 755 faiss_index || true

echo "Build completed successfully!"
