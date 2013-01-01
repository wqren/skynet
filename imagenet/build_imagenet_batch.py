#!/usr/bin/env python

from os.path import basename
import cPickle
import glob
import imagenet
import logging
import mycloud
import os

logging.basicConfig(level=logging.INFO, format="%(created)f %(process)d %(levelname).1s:%(filename)s:%(lineno)3d:%(message)s")

def filename_to_synid(f):
  return basename(f).split('.')[0][1:]

def zip_to_batch():
  os.system('rm "%s"/*' % imagenet.OUTPUTDIR)
  os.system('mkdir -p "%s"' % imagenet.OUTPUTDIR)
  
  batch_meta = { 'label_names' : imagenet.LABEL_NAMES }
  cPickle.dump(batch_meta, open(imagenet.OUTPUTDIR + '/batches.meta', 'w'))

  from mycloud.mapreduce import MapReduce
  from mycloud.resource import Zip, LevelDB, Pickle
  cluster = mycloud.Cluster()

  zips = glob.glob('/hdfs/imagenet/zip/*.zip')
  zips = [f for f in zips if filename_to_synid(f) in imagenet.SYNIDS]
  inputs = [Zip(f) for f in zips]
  outputs = [Pickle(imagenet.OUTPUTDIR + '/batch-%d' % i) for i in range(imagenet.NUM_BATCHES)]

  mr = MapReduce(cluster, 
                 imagenet.imagenet_mapper,
                 imagenet.imagenet_reducer,
                 inputs, outputs)
  mr.run()
      
if __name__ == '__main__':
  zip_to_batch()
  #zip_to_hbase()
