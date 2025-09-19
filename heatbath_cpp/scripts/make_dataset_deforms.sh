#!/bin/bash

BETA=4.0
L=64
Nc=3
./build_Nc${Nc}/src/run_heatbath \
    -b${BETA} -L${L} \
    -n5000000 --n_therm=100000 --n_meas=5000 --n_save=5000 \
    --eps=0.5 --prefix=data/cpn_b${BETA}_L${L}_Nc${Nc} -s650948
