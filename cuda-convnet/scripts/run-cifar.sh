#!/bin/bash

GDB="gdb -ex run --args "
VALGRIND="valgrind"
NUMPROCS=$1

if [[ -z $MINIBATCH ]]; then
  MINIBATCH=32
fi

echo "minibatch: $MINIBATCH"

if [[ -z $1 ]]; then
  echo "Usage: $0 <number of processes>"
  exit 1
fi

mkdir -p output-$NUMPROCS

set -x

 #--gpu=0 \
mpirun\
 -n "$NUMPROCS" \
 -output-filename "$PWD/output-$NUMPROCS/mpi" \
 -hostfile ./hostfile \
  python main.py \
 --data-path=/home/power/datasets/cifar-10-py-colmajor \
 --save-path=/scratch/tmp \
 --test-range=5 \
 --train-range=1-4 \
 --layer-def=./example-layers/cifar.cfg \
 --layer-params=./example-layers/cifar-params.cfg \
 --data-provider=cifar \
 --test-freq=2 \
 --epochs=20 \
 --mini=$MINIBATCH
