#!/bin/bash

# also, the data is normalized to reduce the effect of scale.
# use bs 32 because of the additional scoring branch

# Train bare PCN on ScanNet data
CUDA_VISIBLE_DEVICES=0 python train_scannet_score_cls.py --exp_name ScanNet_score_cls_35 --lr 0.0001 --epochs 40 --batch_size 64 --coarse_loss cd --num_workers 64 --save_frequency 1 --log_frequency 200
