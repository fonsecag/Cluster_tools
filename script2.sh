#!/bin/bash

source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate sgdml032

python run.py cluster_error -d ../Data/datasets/salicylic/salicylic.npz -i ../Data/models/salicylic/salicylic.npz -c clusters/sal_50.npy
wait

python run.py cluster_error -d ../Data/datasets/salicylic/salicylic.npz -i ../Data/models/salicylic/sal_abvmse_weigh_i2_s1.npz -c clusters/sal_50.npy
wait


