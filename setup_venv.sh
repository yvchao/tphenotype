#!/bin/bash 

ENV_NAME="venv"
PYTHON_VERSION="3.9.7"

# set up virtual environment
setup ()
{
  if [[ $CONDA_PREFIX ]]; then
    echo "creating venv via conda"
    conda create --prefix=$ENV_NAME python=$PYTHON_VERSION
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
  fi
  echo "install jupyter lab..."
  source $ENV_NAME/bin/activate
  pip install wheel
  pip install jupyterlab ipywidgets
}

if [[ -d "${ENV_NAME}" && ! -L "${ENV_NAME}" ]]; then
  echo "load existing venv in $ENV_NAME"
else
  setup
fi

# start jupyter lab
source $ENV_NAME/bin/activate
jupyter lab . --port=9999
