#!/usr/bin/env bash
# Download the Corporación Favorita Grocery Sales Forecasting dataset with the Kaggle CLI.
set -euo pipefail
OUT="${1:-data/raw/favorita}"
mkdir -p "$OUT"
kaggle competitions download -c favorita-grocery-sales-forecasting -p "$OUT"
unzip -o "$OUT/favorita-grocery-sales-forecasting.zip" -d "$OUT"
for f in "$OUT"/*.7z; do 7z x -y "$f" -o"$OUT" && rm -f "$f"; done
ls -lh "$OUT"
