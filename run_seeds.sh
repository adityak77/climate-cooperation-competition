#!/bin/bash

for SEED in 100 101 102 103 104; do
    python run_exp.py -s ${SEED}
done
