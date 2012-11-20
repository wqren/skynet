#!/bin/bash

GDB="gdb -ex run --args"

mpirun -n 2 -hostfile hostfile xterm -hold -e gdb -ex run --args python convnet.py \
 --data-path=/big/nn-data/cifar-10-py-colmajor/ \
 --save-path=/big/tmp \
 --test-range=6 \
 --train-range=1-5 \
 --layer-def=./example-layers/layers-conv-local-11pct.cfg \
 --layer-params=./example-layers/layer-params-conv-local-11pct.cfg \
 --data-provider=cifar \
 --test-freq=2 \
 --mini=64
