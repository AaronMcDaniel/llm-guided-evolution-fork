#!/usr/bin/env bash

set -euo pipefail

mkdir -p data

echo "PWD: $(pwd)"
echo "Checking Kaggle access..."
kaggle competitions files -c titanic

echo "Downloading Titanic train.csv..."
kaggle competitions download -c titanic -f train.csv -p data

echo "Files in data/:"
ls -lah data