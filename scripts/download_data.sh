#!/bin/bash
set -e

DATA_DIR="../data"
URL="https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip"
ZIP_FILE="$DATA_DIR/sleep-edf.zip"

mkdir -p $DATA_DIR
echo "Downloading Sleep-EDF dataset..."
wget -O $ZIP_FILE $URL

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DATA_DIR
echo "Done! Files stored in $DATA_DIR."
