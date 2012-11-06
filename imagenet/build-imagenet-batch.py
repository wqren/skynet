#!/usr/bin/env python

from PIL import Image
from os.path import abspath, basename
import cPickle as C
import cStringIO as StringIO
import collections
import functools
import glob
import logging
import mycloud
import numpy as N
import os
import random
import re
import zipfile


class memoized(object):
  def __init__(self, func):
    self.func = func
    self.cache = {}

  def __call__(self, *args):
    assert isinstance(args, collections.Hashable)

    if args in self.cache:
     return self.cache[args]
    else:
     value = self.func(*args)
     self.cache[args] = value
     return value

logging.basicConfig(level=logging.INFO,
                    format="%(created)f - %(levelname).1s:%(filename)s:%(lineno)3d:%(message)s")

DATADIR ='/hdfs/imagenet-zip'
OUTPUTDIR = '/hdfs/imagenet-batch'

batch = {}
logging.info('Initializing syn list')

@memoized
def synids():
  ids = glob.glob(DATADIR + '/*.zip')
  ids = [basename(x)[1:-4] for x in ids]
  return ids

@memoized
def synid_to_name():
  syns = open(DATADIR + '/fall11_synsets.txt').read().split('\n')
  syns = dict([re.split(' ', x, maxsplit=1) for x in syns][:-1])
  for k, v in syns.items():
    syns[k] = v.split(',')[0]
  return syns

@memoized
def label_names():
  return [synid_to_name().get(s, 'unknown') for s in synids()]

cluster = mycloud.Cluster(
    [
      ('beaker-14', 32),
      ('beaker-15', 32),
      ('beaker-16', 32),
      ('beaker-17', 32),
      ('beaker-18', 32),
      ('beaker-19', 32),
      ('beaker-20', 32),
      ('beaker-21', 32),
      ('beaker-22', 32),
      ('beaker-23', 32),
      ('beaker-24', 32),
      ('beaker-25', 32),
   ])

def randsyn():
  ids = synids()
  while 1:
    synid = ids[random.randrange(len(ids))]
    if synid in synid_to_name():
      return synid
  
def get_syn_entry(idx):
  syn = randsyn()
  zipname = DATADIR + '/n%s.zip' % syn
  zf = zipfile.ZipFile(zipname)
  entries = zf.namelist()
  filename = entries[random.randrange(len(entries))]
  data_file = StringIO.StringIO(zf.open(filename, 'r').read())
  img = Image.open(data_file)
  img = img.resize((256, 256), Image.BICUBIC)
  out_bytes = StringIO.StringIO()
  img.save(out_bytes, 'jpeg')
#  logging.info('Finished entry %d', idx)
  return (syn, synid_to_name()[syn], filename, out_bytes.getvalue())

if __name__ == '__main__':
  for batch_num in range(32):
    logging.info('Creating batch: %s', batch_num)
    batch = { 'data' : [], 'filenames' : [], 'names' : [], 'labels' : [] } 
    
    results = cluster.map(get_syn_entry, range(1024))
    for r in results:
      synid, label, filename, data = r
      
      #synid, label, filename, data = get_syn_entry()
      logging.info('%s %s %s', synid, label, filename)
      batch['labels'].append(int(synid))
      batch['names'].append(label)
      batch['data'].append(data)
      batch['filenames'].append(filename)
    
    batch_out = open(OUTPUTDIR + '/data_batch_%d' % batch_num, 'w')
    C.dump(batch, batch_out)
    batch_out.close()
    
  batch_meta = { 'label_names' : label_names() }
  C.dump(batch_meta, open(OUTPUTDIR + '/batches.meta', 'w'))
