#!/bin/bash

# use deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENV_NAME=venv

run_in_venv()
{
    if [[ $CONDA_PREFIX ]]; then
      eval "$(conda shell.bash hook)"
      conda activate ../../$ENV_NAME
      $@
      conda deactivate
    elif [[ $PYENV_ROOT ]]; then
      source ../../$ENV_NAME/bin/activate
      $@
      deactivate
    fi
}

main_exp()
{
    echo  split datasets
    python data_split.py
    echo run hyperparameter selection
    bash ./parameter_selection.sh >>hparams_selection.log 2>&1
    echo run benchmark
    python benchmarks.py >>benchmark.log 2>&1
}

run_in_venv main_exp

echo run external baselines
cd external
bash ./run_benchmark.sh >>external_baselines.log 2>&1

cd ..
run_in_venv python external_benchmarks.py
