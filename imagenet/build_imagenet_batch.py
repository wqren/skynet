#!/usr/bin/env python

import cPickle
import imagenet
import logging
import mycloud
import os
import shelve

logging.basicConfig(level=logging.INFO,
                    format="%(created)f %(process)d %(levelname).1s:%(filename)s:%(lineno)3d:%(message)s")

batch = {}
logging.info('Initializing syn list')

if __name__ == '__main__':
  cluster = mycloud.Cluster(
    [
      ('beaker-14', 6),
      ('beaker-15', 6),
      ('beaker-16', 6),
#      ('beaker-17', 6),
      ('beaker-18', 6),
      ('beaker-19', 6),
      ('beaker-20', 6),
      ('beaker-21', 6),
      ('beaker-22', 6),
      ('beaker-23', 6),
      ('beaker-24', 6),
      ('beaker-25', 6),
   ])
  
  os.system('rm "%s"/*' % imagenet.OUTPUTDIR)
  os.system('mkdir -p "%s"' % imagenet.OUTPUTDIR)
  
  cluster.map(imagenet.build_mini_batch, range(imagenet.NUM_BATCHES))
      
  batch_meta = { 'label_names' : imagenet.LABEL_NAMES }
  cPickle.dump(batch_meta, open(imagenet.OUTPUTDIR + '/batches.meta', 'w'))
