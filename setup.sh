#!/bin/bash
echo "Installing conda dependencies..."
conda install -y -c conda-forge ndcctools

echo "Installing aimsPAX and its pip dependencies..."
pip install .
