#!/bin/bash

python convnet.py \
  --data-path=/big/nn-data/cifar-10-py-colmajor/ \
  --save-path=/big/tmp \
  --test-range=6 \
  --train-range=1-5 \
  --layer-def=./example-layers/layers-conv-local-11pct.cfg \
  --layer-params=./example-layers/layer-params-conv-local-11pct.cfg \
  --data-provider=cifar \
  --test-freq=13 \
  --mini=1
