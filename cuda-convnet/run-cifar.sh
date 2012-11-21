#!/bin/bash

mpirun -n 2 xterm -hold -e gdb -ex run --args python convnet.py \
 --data-path=/home/power/datasets/cifar-10-py-colmajor \
 --save-path=/scratch/tmp \
 --test-range=5 \
 --train-range=1-4 \
 --layer-def=./example-layers/layers-conv-local-11pct.cfg \
 --layer-params=./example-layers/layer-params-conv-local-11pct.cfg \
 --data-provider=cifar \
 --test-freq=2 \
 --mini=256
