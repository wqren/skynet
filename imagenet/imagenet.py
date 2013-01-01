#!/usr/bin/env python

from os.path import basename
import Image
import cStringIO
import glob
import logging
import numpy as N
import random
import re
import zipfile
import cPickle

from PIL import Image

DATADIR = '/hdfs/imagenet/zip'
OUTPUTDIR = '/hdfs/imagenet/batches/imagesize-%d' % IMAGESIZE

NUM_BATCHES = 100
IMAGESIZE = 64

def synids():
  ids = glob.glob(DATADIR + '/*.zip')
  ids = [basename(x)[1:-4] for x in ids]
  return ids

def synid_to_name():
  syns = open(DATADIR + '/fall11_synsets.txt').read().split('\n')
  syns = dict([re.split(' ', x, maxsplit=1) for x in syns][:-1])
  for k, v in syns.items():
    syns[k] = v.split(',')[0]
  return syns

def synid_to_label():
  lines = open('/hdfs/imagenet/synid-to-label').read().strip().split('\n')
  kv = [l.split(' ') for l in lines]
  return dict([(k, int(v)) for k, v in kv])
 
SYNIDS = synids()
SYNID_NAMES = synid_to_name()
SYNIDS = [synid for synid in SYNIDS if synid in SYNID_NAMES]
LABEL_NAMES = [SYNID_NAMES[s] for s in SYNIDS]
SYNID_TO_LABEL = synid_to_label() 

print len(SYNIDS), ' categories.'

def randsyn():
  ids = SYNIDS
  while 1:
    synid = ids[random.randrange(len(ids))]
    if synid in SYNID_NAMES:
      return synid
  
def transform_image(data):
  data_file = cStringIO.StringIO(data)
  img = Image.open(data_file).convert('RGB')
  out = Image.new('RGB', (IMAGESIZE, IMAGESIZE))
  
  thumb = img.copy()
  thumb.thumbnail((IMAGESIZE, IMAGESIZE), Image.BICUBIC)

  x_off = (IMAGESIZE - thumb.size[0]) / 2
  y_off = (IMAGESIZE - thumb.size[1]) / 2
  box = (x_off, y_off, x_off + thumb.size[0], y_off + thumb.size[1])
  out.paste(thumb, box)

  bytes_out = cStringIO.StringIO()
  out.save(bytes_out, 'jpeg')
  return bytes_out.getvalue()

def imagenet_mapper(kv_iter, output):
  for filename, img in kv_iter:
    synid = filename.split('_')[0][1:]
    label_name = SYNID_NAMES[synid]
    label = SYNID_TO_LABEL[synid]
    output(filename, { 
             'data' : transform_image(img), 'label' : label, 'label_name' : label_name, 'synid' : synid })

def imagenet_reducer(kv_iter, output):
  for filename, data in kv_iter:
    output(filename, data) 
