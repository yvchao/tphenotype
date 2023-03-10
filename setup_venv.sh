#!/bin/bash 

ENV_NAME="venv"
PYTHON_VERSION="3.9.7"

install_pkg()
{
  echo "install dependencies..."
  pip install wheel
  #pip install jupyterlab ipywidgets pandas
  pip install -r requirements.txt
  pip install --editable .
}

# set up virtual environment
setup ()
{
  if [[ $CONDA_PREFIX ]]; then
    echo "creating venv via conda"
    eval "$(conda shell.bash hook)"
    conda create --prefix=$ENV_NAME python=$PYTHON_VERSION -y
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
      pyenv install $PYTHON_VERSION
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
  # start jupyter lab
  taskset --cpu-list 0-2 jupyter lab ./notebooks --port=9999
  conda deactivate
elif [[ $PYENV_ROOT ]]; then
  source $ENV_NAME/bin/activate
  # start jupyter lab
  taskset --cpu-list 0-2 jupyter lab ./notebooks --port=9999
  deactivate
fi
