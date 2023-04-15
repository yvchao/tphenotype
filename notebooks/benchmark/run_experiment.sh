#!/bin/bash

# use deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Uncomment the below lines if re-running data splitting and hyperparameter selection:
# rm -rfv ./data/*
# rm -rfv ./hyperparam_selection/*

# Remove the previous benchmark results.
# Uncomment to make sure to re-run all benchmarks from scratch.
# rm -rfv ./benchmark_results/*

# Remove model caches - model loading does not currently work correctly.
rm -rfv ./model_cache/*
rm -rfv ./external/ACTPC/ADNI/*
rm -rfv ./external/ACTPC/ICU/*
rm -rfv ./external/ACTPC/Synth/*
rm -rfv ./external/ACTPC/output/*
rm -rfv ./external/dcn_Seq2Seq/ADNI/*
rm -rfv ./external/dcn_Seq2Seq/ICU/*
rm -rfv ./external/dcn_Seq2Seq/Synth/*
rm -rfv ./external/dcn_Seq2Seq/output/*
# ----------------------------------------------------------------------

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
