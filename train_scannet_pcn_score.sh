#!/bin/bash

# also, the data is normalized to reduce the effect of scale.
# use bs 32 because of the additional scoring branch

# Train bare PCN on ScanNet data
CUDA_VISIBLE_DEVICES=1 python train_scannet_score.py --exp_name ScanNet --lr 0.0001 --epochs 20 --batch_size 32 --coarse_loss cd --num_workers 32 --save_frequency 1 --log_frequency 200
