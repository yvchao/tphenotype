#!/usr/bin/bash

# use deterministic algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "hparams selection for TPhenotype"
echo "encoders"
python hs_encoder.py
echo "predictor"
python hs_predictor.py
# echo "K"
# python hs_K.py
echo "K orig"
python hs_K_orig.py

echo "hparams selection for baselines"
echo "E2Py"
python hs_e2py.py 
echo "E2Pz"
python hs_e2pz.py 
echo "Spectral"
python hs_spectral.py 
