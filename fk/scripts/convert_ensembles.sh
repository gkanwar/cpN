#!/bin/bash

for b in 2.0 2.2 2.4 2.6 2.8 3.0; do
    for L in 16 32 64; do
        python scripts/convert_ensemble.py \
               --in_fname raw_data/cpn_b${b}_Nc2_${L}_${L}_ens.dat \
               --out_fname data/cpn_b${b}_Nc2_${L}_${L}.npy \
               --L=${L} --Nc=2
    done
done
