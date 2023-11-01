#!/bin/bash

# Define the values of k
# k_values=(21 31 34 38 40 41 45 55 103 114)
k_values=(103 110 114 40 41 45 55)
# Iterate through the k values
for k in "${k_values[@]}"
do
    echo "Training with k=$k"
    python train.py --config "configs/dtu.txt" --expname "scan$k" --basedir log_dtu_rescale --datadir "data/rs_dtu_4/DTU/scan$k"
done