#!/bin/bash

# Make sure the script is executable by running: chmod +x run_fed_experiment.sh

# Run the Python script with the specified arguments
python3 ./0_poison_model.py \
  --dataset cifar \
  --model resnet \
  --attack badnet \
  --lr 0.1 \
  --malicious 0.1 \
  --poison_frac 1.0 \
  --local_ep 2 \
  --local_bs 64 \
  --attack_begin 0 \
  --defence avg \
  --epoch 200 \
  --attack_label 5 \
  --attack_goal -1 \
  --trigger 'square' \
  --triggerX 27 \
  --triggerY 27 \
  --gpu 0 \
  --iid 1
