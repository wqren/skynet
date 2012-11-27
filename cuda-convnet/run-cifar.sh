#!/bin/bash

GDB="gdb -ex run --args "
VALGRIND="valgrind"
NUMPROCS=$1

if [[ -z $1 ]]; then
  echo "Usage: $0 <number of processes>"
  exit 1
fi

mkdir -p output-$NUMPROCS

set -x

mpirun\
 -n "$NUMPROCS" \
 -output-filename output-$NUMPROCS/mpi \
 python convnet.py \
 --data-path=/home/power/datasets/cifar-10-py-colmajor \
 --save-path=/scratch/tmp \
 --test-range=5 \
 --train-range=1-4 \
 --layer-def=./example-layers/layers-conv-local-11pct.cfg \
 --layer-params=./example-layers/layer-params-conv-local-11pct.cfg \
 --data-provider=cifar \
 --test-freq=10 \
 --epochs=500 \
 --mini=512
