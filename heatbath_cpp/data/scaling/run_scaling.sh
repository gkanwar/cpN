#!/bin/bash

echo "" > scaling.txt
for t in 1 2 4 8 12 24 36 48 60 72; do
    export OMP_NUM_THREADS=${t}
    time=$(
        srun ../../build/src/run_heatbath -L1024 -b8.0 -e0.5 -n100 -m10 -x10000 -k10 -s1234 -f../tmp/test \
            | tail -n1 | cut -d' ' -f4
        )
    echo "$t $time" >> scaling.txt
done
