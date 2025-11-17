#!/bin/bash
set -e

# Create directories
mkdir -p ../data/raw
cd ../data/raw

# Download dataset (small subset)
URL="https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip"
ZIP_FILE="sleep-edf.zip"

echo "Downloading Sleep-EDF Expanded subset..."
wget -O $ZIP_FILE $URL

echo "Unzipping..."
unzip -q $ZIP_FILE
rm $ZIP_FILE

echo "Dataset ready in data/raw/"

