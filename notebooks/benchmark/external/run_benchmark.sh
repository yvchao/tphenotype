#!/bin/bash
EPOCHS=200

cd ACTPC
python train.py -k 3 -d 'Synth' --epochs=$EPOCHS
python test.py -k 3 -d 'Synth'
python train.py -k 3 -d 'ICU' --epochs=$EPOCHS
python test.py -k 3 -d 'ICU'
python train.py -k 4 -d 'ADNI' --epochs=$EPOCHS
python test.py -k 4 -d 'ADNI'
cd ../dcn_Seq2Seq
python train.py -k 3 -d 'Synth' --epochs=$EPOCHS
python test.py -k 3 -d 'Synth'
python train.py -k 3 -d 'ICU' --epochs=$EPOCHS
python test.py -k 3 -d 'ICU'
python train.py -k 4 -d 'ADNI' --epochs=$EPOCHS
python test.py -k 4 -d 'ADNI'
cd ..
