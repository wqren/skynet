#!/usr/bin/env python

from os.path import basename
import Image
import cStringIO
import glob
import logging
import numpy as N
import os
import random
import re
import shelve
import zipfile

NUM_BATCHES = 48
IMAGESIZE = 32
IMAGES_PER_BATCH = 16384
DATADIR = '/hdfs/imagenet/zip'
OUTPUTDIR = '/hdfs/imagenet/batch-%d' % IMAGESIZE

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

SYNIDS = synids()
SYNID_NAMES = synid_to_name()
LABEL_NAMES = [SYNID_NAMES.get(s, 'unknown') for s in SYNIDS]
IMAGES_PER_SYNID = IMAGES_PER_BATCH / len(SYNIDS)
assert IMAGES_PER_SYNID > 1

def randsyn():
  ids = SYNIDS
  while 1:
    synid = ids[random.randrange(len(ids))]
    if synid in SYNID_NAMES:
      return synid
  
def zip_entry_to_numpy(zf, filename):
  data_file = cStringIO.StringIO(zf.open(filename, 'r').read())
  img = Image.open(data_file)
  img = img.resize((IMAGESIZE, IMAGESIZE), Image.BILINEAR).convert('RGB')
  return N.array(img, dtype=N.single).reshape(IMAGESIZE * IMAGESIZE * 3)

def build_mini_batch(batch_idx):
  batch_data = []
  filenames = []
  names = []
  labels = []
  
  synlist = list(SYNIDS)
  random.shuffle(synlist)
  
  for idx, synid in enumerate(synlist):
    if idx % 50 == 0:
      logging.info('Processing synid: %s, %5d/%5d', synid, idx, len(synlist))
    if not synid in SYNID_NAMES: continue
    label = SYNID_NAMES[synid]
    
    zf = zipfile.ZipFile(DATADIR + '/n%s.zip' % synid)
    entries = zf.namelist()
    
    for _ in range(IMAGES_PER_SYNID):
      filename = entries[random.randrange(len(entries))]
      labels.append(int(synid))
      names.append(label)
      filenames.append(filename)
      batch_data.append(zip_entry_to_numpy(zf, filename))
    
  shelf_file = '/tmp/data_batch_%d' % batch_idx
  os.system('rm %s' % shelf_file)
  s = shelve.open(shelf_file, 'n', protocol=-1, writeback=True)
  s['num_items'] = len(batch_data)
  for idx, data in enumerate(batch_data):
    s['item_%d' % idx] = data
  s['names'] = names
  s['labels'] = labels
  s['filenames'] = filenames
  s.sync()
  s.close()
  
  os.system('cat %s > %s/%s' % (shelf_file, OUTPUTDIR, basename(shelf_file)))


