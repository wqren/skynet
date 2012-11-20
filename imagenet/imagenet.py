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

NUM_BATCHES = 100
IMAGESIZE = 256
OUTPUTDIR = '/hdfs/imagenet/batch-%d' % IMAGESIZE

HBASE_HOST = 'localhost'
HBASE_PORT = 9090

print len(SYNIDS), ' categories.'

def hbase_connect():
  import happybase
  return happybase.Connection(HBASE_HOST)

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
  return N.array(img, dtype=N.int8).reshape(IMAGESIZE * IMAGESIZE * 3)

def zip_to_hbase(synid):
  conn = hbase_connect()
  table = conn.table('imagenet')
  batch = table.batch(batch_size = 512)

  zf = zipfile.ZipFile(DATADIR + '/n%s.zip' % synid)
  entries = zf.namelist()
  for filename in entries:
    data = zf.open(filename, 'r').read()
    img = Image.open(cStringIO.StringIO(data))
    img = img.convert('L')
    batch.put(str(hash(filename)), 
        { 'meta:filename': filename,
          'meta:synid': str(synid),
          'meta:size': str(len(data)),
          'meta:sift128': image_to_sift(img, 128).tostring(),
          'meta:sift64': image_to_sift(img, 64).tostring(),
          'data:image': data })
  batch.send()

def transform_image(data):
  data_file = cStringIO.StringIO(data)
  img = Image.open(data_file)
  img = img.resize((IMAGESIZE, IMAGESIZE), Image.NEAREST).convert('RGB')
  img_out = cStringIO.StringIO()
  img.save(img_out, 'jpeg')
  return img_out.getvalue()

def image_to_sift(img, size):
  img = img.resize((size, size), Image.BICUBIC)
  img = N.array(img, dtype=N.single)
  f, d = vl_sift(img)
  return d

def imagenet_mapper(kv_iter, output):
  for filename, img in kv_iter:
    synid = filename.split('_')[0][1:]
    label_name = SYNID_NAMES[synid]
    label = SYNID_TO_LABEL[synid]
    output(filename, { 
             'data' : transform_image(img), 
             'label' : label, 
             'label_name' : label_name, 
             'synid' : synid })

def imagenet_reducer(kv_iter, output):
  for filename, data in kv_iter:
    output(filename, data) 
