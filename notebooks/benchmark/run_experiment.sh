#!/bin/bash 
echo run hyperparameter selection
bash ./parameter_selection.sh >>hparams_selection.log 2>&1

echo run external baselines
cd external 
bash ./run_benchmark.sh >>external_baselines.log 2>&1

cd ..


