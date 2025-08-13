#!/bin/bash
echo "Installing conda dependencies..."
conda install -y -c conda-forge ndcctools

echo "Installing pip dependencies..."
pip install -r requirements.txt

echo "Installing aimsPAX..."
pip install .