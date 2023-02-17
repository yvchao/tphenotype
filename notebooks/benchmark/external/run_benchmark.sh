#!/bin/bash 

ENV_NAME="venv"
PYTHON_VERSION="3.7.13"
EPOCHS=200

install_pkg()
{
  pip install -r requirements.txt
}

run_exp()
{
  cd ACTPC
  python train.py -k 3 -d 'Synth' --epochs=$EPOCHS
  python test.py -k 3 -d 'Synth'
  python train.py -k 2 -d 'ICU' --epochs=$EPOCHS
  python test.py -k 2 -d 'ICU'
  python train.py -k 4 -d 'ADNI' --epochs=$EPOCHS
  python test.py -k 4 -d 'ADNI'
  cd ../dcn_Seq2Seq
  python train.py -k 3 -d 'Synth' --epochs=$EPOCHS
  python test.py -k 3 -d 'Synth'
  python train.py -k 2 -d 'ICU' --epochs=$EPOCHS
  python test.py -k 2 -d 'ICU'
  python train.py -k 4 -d 'ADNI' --epochs=$EPOCHS
  python test.py -k 4 -d 'ADNI'
  cd ..
}

# set up virtual environment
setup()
{
  if [[ $CONDA_PREFIX ]]; then
    echo "creating venv via conda"
    eval "$(conda shell.bash hook)"
    conda create --prefix=$ENV_NAME python=$PYTHON_VERSION
    conda activate ./$ENV_NAME
    install_pkg
    conda deactivate
  elif [[ $PYENV_ROOT ]]; then
    echo "creating venv via pyenv"
    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

    if [[ -z $(pyenv versions | grep -i $PYTHON_VERSION) ]]; then
      echo "installing python $PYTHON_VERSION..."
    fi
    pyenv shell $PYTHON_VERSION
    python -m venv $ENV_NAME
    pyenv shell --unset
    source $ENV_NAME/bin/activate
    install_pkg
    deactivate
  fi
}

if [[ -d "${ENV_NAME}" && ! -L "${ENV_NAME}" ]]; then
  echo "load existing venv in $ENV_NAME"
else
  setup
fi

if [[ $CONDA_PREFIX ]]; then
  eval "$(conda shell.bash hook)"
  conda activate ./$ENV_NAME
  run_exp
  conda deactivate
elif [[ $PYENV_ROOT ]]; then
  source $ENV_NAME/bin/activate
  run_exp
  deactivate
fi