#!/bin/bash

# Build and deploy documentation for GitHub Pages
# This script builds the Sphinx documentation and copies it to the proper location for GitHub Pages

echo "Building Sphinx documentation..."
make clean
make html

echo "Setting up GitHub Pages..."
# GitHub Pages serves from docs/_build/html/, so we just need to ensure .nojekyll exists
cp _build/html/.nojekyll .

echo "Documentation built successfully!"
echo "GitHub Pages will serve from: docs/_build/html/index.html"
echo "Local file location: $(pwd)/_build/html/index.html"