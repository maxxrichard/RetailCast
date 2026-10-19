#!/usr/bin/env bash
# Download the M5 Forecasting - Accuracy dataset with the Kaggle CLI.
# Requires ~/.kaggle/kaggle.json (https://www.kaggle.com/docs/api) and accepting the competition rules.
set -euo pipefail
OUT="${1:-data/raw/m5}"
mkdir -p "$OUT"
kaggle competitions download -c m5-forecasting-accuracy -p "$OUT"
unzip -o "$OUT/m5-forecasting-accuracy.zip" -d "$OUT"
rm -f "$OUT/m5-forecasting-accuracy.zip"
ls -lh "$OUT"
