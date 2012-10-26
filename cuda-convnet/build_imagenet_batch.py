#!/usr/bin/env python

from PIL import Image
from celery import Celery
from celery.task.control import broadcast
from os.path import abspath, basename
import cPickle as C
import cStringIO as StringIO
import glob
import logging
import numpy as N
import os
import random
import re
import zipfile


BROKER_URL = 'redis://kermit.news.cs.nyu.edu'
BACKEND_URL = 'redis://kermit.news.cs.nyu.edu'

DATADIR = '/hdfs/imagenet-zip'
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
synids = glob.glob(DATADIR + '/*.zip')
synids = [basename(x)[1:-4] for x in synids]

synid_to_name = open(DATADIR + '/fall11_synsets.txt').read().split('\n')
synid_to_name = dict([re.split(' ', x, maxsplit=1) for x in synid_to_name][:-1])
for k, v in synid_to_name.items():
    synid_to_name[k] = v.split(',')[0]

label_names = [synid_to_name.get(s, 'unknown') for s in synids]

def randsyn():
  while 1:
    synid = synids[random.randrange(len(synids))]
    if synid in synid_to_name:
      return synid
  
@celery.task
def get_syn_entry(idx):
  syn = randsyn()
  zipname = DATADIR + '/n%s.zip' % syn
  logging.info('Opening %s', zipname)
  zf = zipfile.ZipFile(zipname)
  entries = zf.namelist()
  filename = entries[random.randrange(len(entries))]
  data_file = StringIO.StringIO(zf.open(filename, 'r').read())
  img = Image.open(data_file)
  img = img.resize((256, 256), Image.BICUBIC)
  out_bytes = StringIO.StringIO()
  img.save(out_bytes, 'jpeg')
  logging.info('Finished entry %d', idx)
  return (syn, synid_to_name[syn], filename, out_bytes.getvalue())
  
if __name__ == '__main__':
  broadcast("pool_restart", arguments={"reload_modules":True})
  batch_meta = {'label_names' : label_names}
  C.dump(batch_meta, open(OUTPUTDIR + '/batches.meta', 'w'))
  
  for batch_num in range(32):
    logging.info('Creating batch: %s', batch_num)
    batch = { 'data' : [],
             'filenames' : [],
             'names' : [],
             'labels' : [] }
    
    results = [get_syn_entry.delay(i) for i in range(1024)]
    for r in results:
      synid, label, filename, data = r.get()
      r.forget()
      
      #synid, label, filename, data = get_syn_entry()
      logging.info('%s %s %s', synid, label, filename)
      batch['labels'].append(int(synid))
      batch['names'].append(label)
      batch['data'].append(data)
      batch['filenames'].append(filename)
    
    batch_out = open(OUTPUTDIR + '/data_batch_%d' % batch_num, 'w')
    C.dump(batch, batch_out)
    batch_out.close()
