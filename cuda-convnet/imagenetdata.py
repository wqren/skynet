# image net data provider

from PIL import Image
from util import pickle,unpickle
import numpy as n
import sys
import signal
from numpy.random import random_integers
from time import time, asctime, localtime, strftime, sleep
from math import *
import threading
from Queue import Queue
from Queue import Empty
import random

MEAN_FILE_EXT = "_mean"

#util functions
def clean_queue( q ):
   try:
       q.get( False )
   except Empty:
       return

def PIL2array(img):
   if img.mode == 'L':
      I = n.asarray( img )
      result = n.zeros( (img.size[1], img.size[0],3 ), n.uint8 )
      result[:,:,0] = I
      result[:,:,1] = I
      result[:,:,2] = I
      return result
   else:
      return n.asarray( img )

def array2PIL(arr):
   return Image.fromarray( n.uint8(arr) )

# load image from file
class ReadImage( threading.Thread ):
    def __init__( self,
                  raw_image_queue,  # shared Queue to store raw image
                  data,             # data file contians image path information
                  mean_file,        # mean file
                  root_path,        # root path of images
                  data_mode,        # 'all','train','val'
                  batch_size = 128, # size of batch
                  batch_index = 0,  # start batch index
                  epoch_index = 1   # start epoch index
                  ):
        threading.Thread.__init__( self, name = "Load Image Thread" )
        self.stop = False
        self.sharedata = raw_image_queue
        self.data = data
        self.num_classes = len(self.data['val'])
        self.data_mode = data_mode
        self.root_path = root_path
        if data_mode == "val":
           self.images = self.data['val']
           self.total_samples = self.data['num_data_val']
           self.shuffle = False
           print 'Validation data is not randomized'
        elif data_mode == "train":
           self.images = self.data['train']
           self.total_samples = self.data['num_data_train']
           self.shuffle = True
           print 'Traing data is randomized'
        else:
           print "data_mode: " + str(data_mode) + " not valid"
           import pdb; pdb.set_trace()
           sys.exit(1)
        # iterator on classes
        self.iclass = -1
        # iterator for samples of each class
        self.isamples = self.num_classes * [-1]

        # class_iter = range(num_classes)
        # if shuffle: random.shuffle(class_iter)
        # classes_iter = []
        # for i in range(num_classes):
        #    classes_iter.append(range(len(images[i])))
        #    if shuffle: random.shuffle(classes_iter[i])

        # # get batch queue
        # self.batch_queue = []
        # has_add = True
        # while has_add:
        #    has_add = False
        #    for i in range( self.num_classes ):
        #       if len(index_map[i]) > 0:
        #          index = index_map[i].pop()
        #          self.batch_queue.append( index )
        #          has_add = True

        # self.num_images = len( self.batch_queue )

        #init current index and batch size
        self.total_processed = 0
        self.batch_size = batch_size
        self.batch_index = batch_index
        self.epoch_index = epoch_index
        # read data mean from file
        data_mean_file = unpickle(mean_file)
        self.data_mean = data_mean_file['data']
        # store it as uint8
        self.data_mean = n.round( self.data_mean).astype(n.uint8)
        print data_mode + ': total_samples: ' + str(self.total_samples) \
            + ' batch_size: ' + str(batch_size) \
            + ' num_batches: ' + str(self.get_num_batches())

    def stopThread( self ):
        self.stop = True

    def run( self ):
        while not self.stop:
            data = self.produceData()
            self.sharedata.put( data )

    def produceData( self ):
       epoch_index2 = 1 + self.total_processed / self.total_samples
       if epoch_index2 != self.epoch_index:
          self.epoch_index = epoch_index2
          self.batch_index = 0
       images = []
       labels = []
       for i in range( self.batch_size ):
           image_i,label_i = self.next()
           images.append( image_i )
           labels.append( label_i )
       self.batch_index += 1
       # print 'batch ' + str(self.batch_index) + ' loaded, total processed ' \
       #     + str(self.total_processed)
       return (self.epoch_index, self.batch_index, images, labels)

    def next(self):
       if self.shuffle: # class and sample selection are random
          self.iclass = int(random.uniform(0, self.num_classes))
          self.isamples[self.iclass] = int(random.uniform(0, len(self.images[self.iclass])))
       else: # not random
          self.iclass += 1
          if self.iclass == self.num_classes: self.iclass = 0
          self.isamples[self.iclass] += 1
          if self.isamples[self.iclass] == len(self.images[self.iclass]):
             self.isamples[self.iclass] = 0
       self.total_processed += 1
       image_path = self.root_path + "/" + self.images[self.iclass][self.isamples[self.iclass]]
       # print 'loading class ' + str(self.iclass) \
       #     + ' / sample ' + str(self.isamples[self.iclass]) + ': ' + image_path
       im = Image.open(image_path)
       image_matrix = PIL2array( im )
       return image_matrix,self.iclass

    def get_num_batches( self ):
        return int(ceil(self.total_samples / self.batch_size))
    # int(ceil( 1.0 * len(self.batch_queue) / self.batch_size ))

    def get_num_classes( self ):
        return self.num_classes

    def print_data_summary( self ):
        label_hist = [0] * self.get_num_classes()
        total = 0
        for i in range(len(self.images)):
           label_hist[i] += len(self.images[i])
           total += len(self.images[i])
        print "#samples: " + str(total)
        print "Class Label Hist: ", label_hist, len(label_hist)
        #print "Num Batches     : ", self.get_num_batches()


# read image from raw_image_queue and store to batch_data_queue
class ProcessImage( threading.Thread ):
    def __init__( self,
                  data,
                  raw_image_queue,        #[in]  queue to store ( epoch, batch_index, image )
                  batch_data_queue,       #[out] queue to store transformed batches
                  crop_width,             #[in]  crop width
                  crop_height,            #[in]  crop height
                  data_mean,              #[in]  data mean matrix
                  data_mode,
                  random_transform = True #[in]  whether apply transformation
                  ):
        threading.Thread.__init__( self, name = "Image Process Thread" )
        self.classes = data['classes']
        self.raw_image_queue = raw_image_queue
        self.batch_data_queue = batch_data_queue
        self.random_transform = random_transform
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.data_mean = data_mean
        self.data_mode = data_mode
        self.stop = False

    def stopThread( self ):
        self.stop = True

    def run( self ):
        while not self.stop:
            data = self.raw_image_queue.get()
            data = self.process_image( data )
            self.batch_data_queue.put( data )

    def process_image( self, data ):
        epoch = data[0]
        batch = data[1]
        images = data[2]
        labels = data[3]
        assert( len(images) == len(labels) )
        num_images = len( images )
        result_data = n.zeros( ( self.crop_width * self.crop_height * 3,
            num_images ), n.float32 )
        result_label = n.zeros( (1, num_images ), n.float32 )

        for i in range( num_images ):
            image_matrix = images[i]
            image_matrix = image_matrix.astype(n.float32)
            image_matrix = image_matrix - self.data_mean
            #image_matrix = images[i].astype(n.float32)

            x = 0
            y = 0
            (w,h,a) = image_matrix.shape
            # compute image_matrix
            if self.random_transform:
                # random crop
                x += random_integers( 0, w - self.crop_width - 1)
                y += random_integers( 0, h - self.crop_height - 1)
            else:
                # fixed crop
                x += (w - self.crop_width)/2
                y += (h - self.crop_height)/2

            #crop image
            assert( x + self.crop_width < w )
            assert( y + self.crop_height < h )
            #im = im.crop( (x,y, x + self.crop_width, y + self.crop_height ) )
            image_matrix = image_matrix[ x:x+self.crop_width, y:y+self.crop_width, : ]

            if self.random_transform:
                # flip: roll a dice to whether flip image
                if random_integers( 0,1 ) > 0.5:
                    #im = im.transpose( Image.FLIP_LEFT_RIGHT )
                    image_matrix = image_matrix[:, -1::-1, :]

            image_matrix = image_matrix.reshape( (self.crop_width * self.crop_height * 3, ) )

            #store to result_data
            result_data[:,i] = image_matrix;
            result_label[0,i] = labels[i]

        #return process tuple
        return epoch, batch, result_data, result_label

# main class

# Imagenet data provider
class ImagenetDataProvider:
   def __init__(self, 
                root_path, 
                batch_range, 
                init_epoch=1, 
                init_batchnum=None, 
                dp_params={}, 
                test=False,
                data_mode = "train",
                batch_index = 0,
                epoch_index = 1,
                random_transform = False,
                batch_size = 128,
                crop_width = 224,
                crop_height = 224,
                buffer_size = 2):

       mean_file = root_path + '/python/mean1000.pickle'
       data_file = root_path + '/python/data1000.pickle'

       # init data Q
       self.raw_image_queue = Queue( buffer_size )
       self.batch_data_queue = Queue( buffer_size )
       self.stop = False
       print 'Loading data from ' + str(data_file)
       self.data = unpickle(data_file)
       
       self.readImage = ReadImage(self.raw_image_queue, self.data, mean_file, root_path, data_mode, batch_size, batch_index, epoch_index )
       self.processImage = ProcessImage(self.data, self.raw_image_queue, self.batch_data_queue, crop_width, crop_height, self.readImage.data_mean, data_mode, random_transform)
       
       self.readImage.start()
       self.processImage.start()

   def get_data_dims( self, idx ):
      if idx == 0:
         return self.processImage.crop_width * self.processImage.crop_height * 3
      if idx == 1:
         return 1

   def get_next_batch( self ):
       # construct next batch online
       # batch_data[0]: epoch
       # batch_data[1]: batchnum
       # batch_data[2]['label']: each column represents an image
       # batch_data[2]['data'] : each column represents an image
       # this function only crop center 256 x 256 in image for classification
       data = self.batch_data_queue.get()
       epoch_index = data[0]
       batch_index = data[1]

       result = {}
       result['data'] = data[2]
       result['label'] = data[3]
       #import pdb; pdb.set_trace()
       return epoch_index, batch_index, (data[2], data[3])

   def get_num_classes( self ):
       return self.readImage.num_classes

   def get_num_batches( self ):
       return self.readImage.get_num_batches()

   def print_data_summary( self ):
       return self.readImage.print_data_summary()

   def stopThread( self ):
       self.stop = True

       self.processImage.stopThread()
       # make sure processImage thread should not
       # be blocked by get method of raw_image_queue
       while self.raw_image_queue.empty():
           sleep( 1 )

       # make sure processImage thread should not
       # be blocked by put method of batch_data_queue
       while not self.batch_data_queue.empty():
           clean_queue( self.batch_data_queue )

       self.readImage.stopThread()
       # make sure readImage thread should not
       # be blocked by put method of raw_image_queue
       while not self.raw_image_queue.empty():
           clean_queue( self.raw_image_queue )

       self.readImage.join()
       self.processImage.join()
