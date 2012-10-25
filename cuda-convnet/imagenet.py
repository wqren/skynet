#!/usr/bin/env python

from PIL import Image
from os.path import abspath, basename
import cPickle as C
import glob
import logging
import os
import random
import re
import tarfile

from celery import Celery

BROKER_URL = 'redis://kermit.news.cs.nyu.edu'
BACKEND_URL = 'redis://kermit.news.cs.nyu.edu'

DATADIR = '/hdfs/imagenet'
OUTPUTDIR = '/big/nn-data/imagenet-sample'

#if not os.path.exists(OUTPUTDIR):
#  os.mkdir(OUTPUTDIR)

celery = Celery('imagenet',
                backend=BACKEND_URL, 
                broker=BROKER_URL)


logging.basicConfig(level=logging.INFO,
                    format="%(created)f - %(levelname).1s:%(filename)s:%(lineno)3d:%(message)s")

batch = {}
logging.info('Initializing syn list')
syns = glob.glob(DATADIR + '/*.tar')
syns = [basename(x)[1:-4] for x in syns]

synid_to_name = open(DATADIR + '/fall11_synsets.txt').read().split('\n')
synid_to_name = dict([re.split(' ', x, maxsplit=1) for x in synid_to_name][:-1])
for k, v in synid_to_name.items():
    synid_to_name[k] = v.split(',')[0]

def randsyn():
  while 1:
    synid = syns[random.randrange(len(syns))]
    if synid in synid_to_name:
      return synid
  
@celery.task
def get_syn_entry(idx):
  syn = randsyn()
  tf = tarfile.open(DATADIR + '/n' + syn + '.tar')
  entries = tf.getnames()
  filename = entries[random.randrange(len(entries))]
  data_file = tf.extractfile(filename)
  data = data_file.read()
  logging.info('Finished entry %d', idx)
  return (syn, synid_to_name[syn], filename, data)
  
if __name__ == '__main__':
  for batch_num in range(10):
    logging.info('Creating batch: %s', batch_num)
    batch = { 'data' : [],
             'filenames' : [],
             'names' : [],
             'labels' : [] }
    
    results = [get_syn_entry.delay(i) for i in range(1024)]
    for r in results:
      synid, label, filename, data = r.get()
      
      #synid, label, filename, data = get_syn_entry()
      logging.info('%s %s %s', synid, label, filename)
      batch['labels'].append(int(synid))
      batch['names'].append(label)
      batch['data'].append(data)
      batch['filenames'].append(filename)
    
    batch_out = open(OUTPUTDIR + '/batch-%d' % batch_num, 'w')
    C.dump(batch, batch_out)
    batch_out.close()
