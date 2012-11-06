#!/usr/bin/env python
from os.path import basename
import Image
import cStringIO
import glob
import logging
import random
import re
import zipfile

IMAGESIZE = 128
DATADIR ='/hdfs/imagenet/zip'
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

def randsyn():
  ids = SYNIDS
  while 1:
    synid = ids[random.randrange(len(ids))]
    if synid in SYNID_NAMES:
      return synid
  
def get_syn_entry(idx):
  syn = randsyn()
  zipname = DATADIR + '/n%s.zip' % syn
  zf = zipfile.ZipFile(zipname)
  entries = zf.namelist()
  filename = entries[random.randrange(len(entries))]
  data_file = cStringIO.StringIO(zf.open(filename, 'r').read())
  img = Image.open(data_file)
  img = img.resize((IMAGESIZE, IMAGESIZE), Image.BILINEAR)
  out_bytes = cStringIO.StringIO()
  img.save(out_bytes, 'jpeg')
#  logging.info('Finished entry %d', idx)
  return (syn, SYNID_NAMES[syn], filename, out_bytes.getvalue())

def profile_syn_entry(idx):
  z = get_syn_entry
  import cProfile 
  import pstats
  
  prof = cProfile.Profile()
  prof.runctx('get_syn_entry(idx)', globals(), locals())
  prof_out = cStringIO.StringIO()
  pstats.Stats(prof, stream=prof_out).strip_dirs().sort_stats(-1).print_stats()
  
  logging.info('%s', prof_out.getvalue())


