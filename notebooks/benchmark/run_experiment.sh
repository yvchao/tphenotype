#!/bin/bash 
echo run hyperparameter selection

if [[ $CONDA_PREFIX ]]; then
  eval "$(conda shell.bash hook)"
  conda activate ../../$ENV_NAME
  
  python data_split.py
  bash ./parameter_selection.sh >>hparams_selection.log 2>&1
  
  conda deactivate
elif [[ $PYENV_ROOT ]]; then
  source ../../$ENV_NAME/bin/activate
  
  python data_split.py
  bash ./parameter_selection.sh >>hparams_selection.log 2>&1
  
  deactivate
fi


echo run external baselines
cd external 
bash ./run_benchmark.sh >>external_baselines.log 2>&1

cd ..


