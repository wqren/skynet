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
mkdir -p /scratch/$USER

set -x

mpirun\
 --mca btl_tcp_if_exclude virbr0,eth1 \
 -n "$NUMPROCS" \
 -output-filename "$PWD/output-$NUMPROCS/mpi" \
 -hostfile ./hostfile \
 python main.py \
 --data-path=/home/snwiz/data/imagenet12/ \
 --save-path=./scratch/power/imagenet-test \
 --layer-def=./imagenet-layers/layers-imagenet.cfg \
 --layer-params=./imagenet-layers/layer-params-imagenet2.cfg \
 --train-range=1-10000 \
 --test-range=10001 \
 --test-freq=10 \
 --exchange-freq=10 \
 --data-provider=imagenet \
 --epochs=10 \
 --mini=16

