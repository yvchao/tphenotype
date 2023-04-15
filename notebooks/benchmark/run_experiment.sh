#!/bin/bash

# use deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Uncomment the below two lines to re-run data splitting and hyperparameter selection:
# rm -rfv ./data/*
# rm -rfv ./hyperparam_selection/*

echo  "Split datasets..."
python -u data_split.py
echo "Run hyperparameter selection..."
bash ./parameter_selection.sh > hparams_selection.log 2>&1
echo "Run benchmarks..."
python -u benchmarks.py > benchmark.log 2>&1

echo "Run external baselines..."
cd external
bash ./run_benchmark.sh > external_baselines.log 2>&1

cd ..
python -u external_benchmarks.py
