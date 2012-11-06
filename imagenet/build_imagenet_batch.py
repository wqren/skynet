#!/usr/bin/env python

from imagenet import OUTPUTDIR, SYNID_NAMES, SYNIDS, get_syn_entry
import cPickle
import logging
import mycloud
import os

LABEL_NAMES = [SYNID_NAMES.get(s, 'unknown') for s in SYNIDS]

logging.basicConfig(level=logging.INFO,
                    format="%(created)f - %(levelname).1s:%(filename)s:%(lineno)3d:%(message)s")

batch = {}
logging.info('Initializing syn list')

if __name__ == '__main__':
  cluster = mycloud.Cluster(
    [
      ('beaker-14', 64),
      ('beaker-15', 64),
      ('beaker-16', 64),
      ('beaker-17', 64),
      ('beaker-18', 64),
      ('beaker-19', 64),
      ('beaker-20', 64),
      ('beaker-21', 64),
      ('beaker-22', 64),
      ('beaker-23', 64),
      ('beaker-24', 64),
      ('beaker-25', 64),
   ])
  
  os.system('rm "%s/*"' % OUTPUTDIR)
  os.system('mkdir -p "%s"' % OUTPUTDIR)

  for batch_num in range(32):
    logging.info('Creating batch: %s', batch_num)
    batch = { 'data' : [], 'filenames' : [], 'names' : [], 'labels' : [] } 
    
    results = cluster.map(get_syn_entry, range(1024))
    for r in results:
      synid, label, filename, data = r
      
      #synid, label, filename, data = get_syn_entry()
      #logging.info('%s %s %s', synid, label, filename)
      batch['labels'].append(int(synid))
      batch['names'].append(label)
      batch['data'].append(data)
      batch['filenames'].append(filename)
    
    batch_out = open(OUTPUTDIR + '/data_batch_%d' % batch_num, 'w')
    cPickle.dump(batch, batch_out)
    batch_out.close()
    
  batch_meta = { 'label_names' : LABEL_NAMES }
  cPickle.dump(batch_meta, open(OUTPUTDIR + '/batches.meta', 'w'))
