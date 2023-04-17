#!/usr/bin/bash

# use deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "hparams selection for TPhenotype"
echo "encoders"
python -u hs_encoder.py
echo "K orig"
python -u hs_K_orig.py

echo "hparams selection for baselines"
echo "E2Py"
python -u hs_e2py.py
echo "E2Pz"
python -u hs_e2pz.py
echo "Spectral"
python -u hs_spectral.py
