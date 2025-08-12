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

#!/bin/bash

# Build script for Render deployment
set -e

echo "Starting build process..."
echo "Python version:"
python3 --version

# Upgrade pip to latest version
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Set environment variables for better builds
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

# Install requirements with verbose output
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt --verbose

echo "Build completed successfully!"

# Set proper permissions
chmod -R 755 uploads || true
chmod -R 755 faiss_index || true

echo "Build completed successfully!"
