#!/bin/bash
echo "Installing conda dependencies..."
conda install -y -c conda-forge ndcctools

echo "Installing Python package..."
pip install .
