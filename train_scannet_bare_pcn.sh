#!/bin/bash

# also, the data is normalized to reduce the effect of scale.

# Train bare PCN on ScanNet data
CUDA_VISIBLE_DEVICES=2 python train_scannet.py --exp_name ScanNet_35 --lr 0.0001 --epochs 40 --batch_size 64 --coarse_loss cd --num_workers 64 --save_frequency 1 --log_frequency 200
