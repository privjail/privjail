#!/bin/bash
set -euo pipefail

mkdir -p data
cd data/

curl -L -o adult.zip https://www.kaggle.com/api/v1/datasets/download/wenruliu/adult-income-dataset
unzip adult.zip

test_samples=16281
train_samples=32561
head -n $(( test_samples + 1 )) adult.csv > adult_test.csv
head -n 1 adult.csv > adult_train.csv
tail -n $(( train_samples )) adult.csv >> adult_train.csv
