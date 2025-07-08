#!/bin/bash
set -euo pipefail

mkdir -p data
cd data/

# curl -L -o tdriver.zip https://www.kaggle.com/api/v1/datasets/download/arashnic/tdriver
# unzip tdriver.zip
duckdb -f ../concat_tdrive.sql
