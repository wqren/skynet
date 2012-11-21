#!/bin/bash

#gdb -ex run --args 
python convnet.py \
  --data-path=/home/power/imagenet/batch-64 \
  --save-path=/scratch/power/tmp \
  --gpu=0 \
  --test-range=11 \
  --train-range=1-10 \
  --layer-def=./imagenet-64.cfg \
  --layer-params=./imagenet-params.cfg \
  --data-provider=imagenet \
  --test-freq=13 \
  --mini=512
