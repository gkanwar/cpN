#!/bin/bash

echo "" > scaling.txt
for t in 1 2 4 8 12 24 36 48 60 72; do
    export OMP_NUM_THREADS=${t}
    time=$(
        srun ../../build/src/run_heatbath -L1024 -b8.0 --eps=0.5 --n_iter=100 --n_meas=10 --n_save=10000 -s1234 --prefix=../tmp/test \
            | tail -n1 | cut -d' ' -f4
        )
    echo "$t $time" >> scaling.txt
done
