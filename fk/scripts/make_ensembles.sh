#!/bin/bash

set +x

NC=2
BETA_NS=(2.0 2.2 2.4 2.6 2.8 3.0)
LS=(16 32 64)

BIN="../heatbath_cpp/build_Nc${NC}/src/run_heatbath"
if [[ ! -f ${BIN} ]]; then
    echo "Could not find binary ${BIN}"
    exit 1
fi

SEED=8123610000
i=0
for L in "${LS[@]}"; do
    for BETA_N in "${BETA_NS[@]}"; do
        SEED_I=$((SEED+i))
        OUT_PREFIX="cpn_b${BETA_N}_Nc${NC}_${L}_${L}"
        ${BIN} --beta=${BETA_N} --L=${L} --n_therm=10000 --n_meas=10 \
               --n_save=100 --n_iter=1000000 --eps=0.25 \
               --seed=${SEED_I} \
               --prefix=raw_data/${OUT_PREFIX} \
               > raw_data/${OUT_PREFIX}.stdout \
               2> raw_data/${OUT_PREFIX}.stderr &
        ((i++))
    done
    wait
done
