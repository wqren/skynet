# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from convdata import *
from data import *
from gpumodel import *
from mpi4py import MPI
from options import *
from os import linesep as NL
from time import time, asctime, strftime, localtime
from util import *

import convnet
import ctypes
import layer as lay
import math as m
import numpy as n
import numpy.random as nr
import sys



#import pylab as pl

MPILIB = ctypes.CDLL('libmpi.so', ctypes.RTLD_GLOBAL)
PALLIB = ctypes.CDLL('libopen-pal.so', ctypes.RTLD_GLOBAL)


class ConvNetRunner(object):
    def __init__(self, op, load_dic, dp_params={}):
        filename_options = []
        dp_params['multiview_test'] = op.get_value('multiview_test')
        dp_params['crop_border'] = op.get_value('crop_border')

        # these are input parameters
        self.model_name = 'ConvNet'
        self.op = op
        self.options = op.options
        self.load_dic = load_dic
        self.filename_options = filename_options
        self.dp_params = dp_params
        self.get_gpus()
        self.fill_excused_options()

        #assert self.op.all_values_given()
        
        for o in op.get_options_list():
            setattr(self, o.name, o.value)
        
        n.random.shuffle(self.train_batch_range)

        # these are things that the model must remember but they're not input parameters
        if load_dic:
            self.model_state = load_dic["model_state"]
            self.save_file = self.options["load_file"].value
            if not os.path.isdir(self.save_file):
                self.save_file = os.path.dirname(self.save_file)
        else:
            self.model_state = {}
            if filename_options is not None:
                self.save_file = 'ConvNet' + "_" + '_'.join(['%s_%s' % (char, self.options[opt].get_str_value()) for opt, char in filename_options]) + '_' + strftime('%Y-%m-%d_%H.%M.%S')
            self.model_state["train_outputs"] = []
            self.model_state["test_outputs"] = []
            self.model_state["epoch"] = 1
            self.model_state["batchnum"] = self.train_batch_range[0]

        self.init_data_providers()
        if load_dic: 
            self.train_data_provider.advance_batch()
            
        # model state often requries knowledge of data provider, so it's initialized after
        try:
            self.init_model_state()
        except ModelStateException, e:
            print e
            sys.exit(1)
        
        for var, val in self.model_state.iteritems():
            setattr(self, var, val)
            
        self.init_model_lib()

    def init_data_providers(self):
        self.dp_params['convnet'] = self
        try:
            self.test_data_provider = DataProvider.get_instance(self.data_path, self.test_batch_range,
                                                                type=self.dp_type, dp_params=self.dp_params, test=True)
            self.train_data_provider = DataProvider.get_instance(self.data_path, self.train_batch_range,
                                                                     self.model_state["epoch"], self.model_state["batchnum"],
                                                                     type=self.dp_type, dp_params=self.dp_params, test=False)
        except DataProviderException, e:
            print "Unable to create data provider: %s" % e
            self.print_data_providers()
            sys.exit(1)

    def start(self):
        if self.test_only:
            self.test_outputs += [self.get_test_error()]
            self.print_test_results()
            sys.exit(1)
        self.train()
    
    def train(self):
        print "========================="
        print "Training %s" % self.model_name
        self.op.print_values()
        print "========================="
        self.print_model_state()
        print "Running on CUDA device(s) %s" % ", ".join("%d" % d for d in self.device_ids)
        print "Current time: %s" % asctime(localtime())
        print "Saving checkpoints to %s" % os.path.join(self.save_path, self.save_file)
        print "========================="
        next_data = self.get_next_batch()
        while self.epoch <= self.num_epochs:
            data = next_data
            self.epoch, self.batchnum = data[0], data[1]
            self.print_iteration()
            sys.stdout.flush()
            
            compute_time_py = time()
            self.start_batch(data)
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch()
            
            batch_output = self.finish_batch()
            self.train_outputs += [batch_output]
            self.print_train_results()

            if self.get_num_batches_done() % self.testing_freq == 0:
                self.sync_with_host()
                self.test_outputs += [self.get_test_error()]
                self.print_test_results()
                self.print_test_status()
                self.conditional_save()
            
            self.print_train_time(time() - compute_time_py)
        self.cleanup()
    
    def print_model_state(self):
        pass
    
    def cleanup(self):
        sys.exit(0)
    
    def sync_with_host(self):
        worker = convnet.SyncWorker(self.model)
        worker.thisown = 0
        self.model.getWorkerQueue().enqueue(worker)
        res = self.model.getResultQueue().dequeue()
        assert res.getResultType() == convnet.WorkResult.SYNC_DONE

            
    def get_num_batches_done(self):
        return len(self.train_batch_range) * (self.epoch - 1) + self.batchnum - self.train_batch_range[0] + 1
    
    def get_next_batch(self, train=True):
        dp = self.train_data_provider
        if not train:
            dp = self.test_data_provider

        batch = dp.get_next_batch()
        return self.parse_batch_data(batch, train=train)
    
    def finish_batch(self):
        result = self.model.getResultQueue().dequeue()
        cost = result.getResults()
        return cost.getCostMap(), cost.getNumCases()
    
    def conditional_save(self):
        batch_error = self.test_outputs[-1][0]
        if batch_error > 0 and batch_error < self.max_test_err:
            self.save_state()
        else:
            print "\tTest error > %g, not saving." % self.max_test_err,
    
    def aggregate_test_outputs(self, test_outputs):
        test_error = tuple([sum(t[r] for t in test_outputs) / (1 if self.test_one else len(self.test_batch_range)) for r in range(len(test_outputs[-1]))])
        return test_error
    
    def get_test_error(self):
        next_data = self.get_next_batch(train=False)
        test_outputs = []
        while True:
            data = next_data
            self.start_batch(data, train=False)
            load_next = not self.test_one and data[1] < self.test_batch_range[-1]
            if load_next: # load next batch
                next_data = self.get_next_batch(train=False)
            test_outputs += [self.finish_batch()]
            if self.test_only: # Print the individual batch results for safety
                print "batch %d: %s" % (data[1], str(test_outputs[-1]))
            if not load_next:
                break
            sys.stdout.flush()
            
        return self.aggregate_test_outputs(test_outputs)
    
    def set_var(self, var_name, var_val):
        setattr(self, var_name, var_val)
        self.model_state[var_name] = var_val
        return var_val
        
    def get_var(self, var_name):
        return self.model_state[var_name]
        
    def has_var(self, var_name):
        return var_name in self.model_state
        
    def save_state(self):
        for att in self.model_state:
            if hasattr(self, att):
                self.model_state[att] = getattr(self, att)
        
        dic = {"model_state": self.model_state,
               "op": self.op}
            
        checkpoint_dir = os.path.join(self.save_path, self.save_file)
        checkpoint_file = "%d.%d" % (self.epoch, self.batchnum)
        checkpoint_file_full_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        pickle(checkpoint_file_full_path, dic,compress=self.zip_save)
        
        for f in sorted(os.listdir(checkpoint_dir), key=alphanum_key):
            if sum(os.path.getsize(os.path.join(checkpoint_dir, f2)) for f2 in os.listdir(checkpoint_dir)) > self.max_filesize_mb*1024*1024 and f != checkpoint_file:
                os.remove(os.path.join(checkpoint_dir, f))
            else:
                break
            
    @staticmethod
    def load_checkpoint(load_dir):
        if os.path.isdir(load_dir):
            return unpickle(os.path.join(load_dir, sorted(os.listdir(load_dir), key=alphanum_key)[-1]))
        return unpickle(load_dir)

    @staticmethod
    def print_data_providers():
        print "Available data providers:"
        for dp, desc in dp_types.iteritems():
            print "    %s: %s" % (dp, desc)
            
    @staticmethod
    def parse_options(op):
        try:
            load_dic = None
            options = op.parse()
            if options["load_file"].value_given:
                load_dic = IGPUModel.load_checkpoint(options["load_file"].value)
                old_op = load_dic["op"]
                old_op.merge_from(op)
                op = old_op
            op.eval_expr_defaults()
            return op, load_dic
        except OptionMissingException, e:
            print e
            op.print_usage()
        except OptionException, e:
            print e
        except UnpickleError, e:
            print "Error loading checkpoint:"
            print e
        sys.exit(1)

    def init_model_lib(self):
        num_peers = MPI.COMM_WORLD.Get_size()
        
        # Adjust our learning rate based on the number of other workers - more workers -> larger
        # batch size -> higher learning rate.
        print 'Resetting model parameters for %d peers.' % num_peers
        rate_adjustment = num_peers / 32.
        for l in self.layers:
          pass
          #if 'epsW' in l: l['epsW'] = [rate_adjustment * num_peers for w in l['epsW']]
          #if 'epsB' in l: l['epsB'] = rate_adjustment * num_peers
          #if 'momW' in l: l['momW'] = [w / num_peers for w in l['momW']]
          #if 'momB' in l: l['momB'] = w / num_peers

        self.model = convnet.ConvNet(self.layers, self.minibatch_size, self.device_ids[0])
        self.model.start()
        
    def init_model_state(self):
        ms = self.model_state
        if self.load_file:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
        else:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self)
        self.layers_dic = dict(zip([l['name'] for l in ms['layers']], ms['layers']))
        
        logreg_name = self.op.get_value('logreg_name')
        if logreg_name:
            self.logreg_idx = self.get_layer_idx(logreg_name, check_type='cost.logreg')
        
        # Convert convolutional layers to local
        if len(self.op.get_value('conv_to_local')) > 0:
            for i, layer in enumerate(ms['layers']):
                if layer['type'] == 'conv' and layer['name'] in self.op.get_value('conv_to_local'):
                    lay.LocalLayerParser.conv_to_local(ms['layers'], i)
        # Decouple weight matrices
        if len(self.op.get_value('unshare_weights')) > 0:
            for name_str in self.op.get_value('unshare_weights'):
                if name_str:
                    name = lay.WeightLayerParser.get_layer_name(name_str)
                    if name is not None:
                        name, idx = name[0], name[1]
                        if name not in self.layers_dic:
                            raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                        layer = self.layers_dic[name]
                        lay.WeightLayerParser.unshare_weights(layer, ms['layers'], matrix_idx=idx)
                    else:
                        raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
        self.op.set_value('conv_to_local', [], parse=False)
        self.op.set_value('unshare_weights', [], parse=False)

    
    def get_layer_idx(self, layer_name, check_type=None):
        try:
            layer_idx = [l['name'] for l in self.model_state['layers']].index(layer_name)
            if check_type:
                layer_type = self.model_state['layers'][layer_idx]['type']
                if layer_type != check_type:
                    raise ModelStateException("Layer with name '%s' has type '%s'; should be '%s'." % (layer_name, layer_type, check_type))
            return layer_idx
        except ValueError:
            raise ModelStateException("Layer with name '%s' not defined." % layer_name)

    def fill_excused_options(self):
        if self.op.get_value('check_grads'):
            self.op.set_value('save_path', '')
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Make sure the data provider returned data in proper format
    def parse_batch_data(self, batch_data, train=True):
        if max(d.dtype != n.single for d in batch_data[2]):
            raise DataProviderException("All matrices returned by data provider must consist of single-precision floats.")
        return batch_data

    def get_gpus(self):
        rank = int(MPI.COMM_WORLD.Get_rank())
        gpus = self.op.get_value('gpu')
        self.device_ids = [gpus[rank % gpu_count()]]
        if self.device_ids[0] == -1:
          self.device_ids[0] = rank % gpu_count()
        print >>sys.stderr, 'MPI RANK: %d, device %s' % (rank, self.device_ids)

    def start_batch(self, batch_data, train=True):
        data = batch_data[2]
        cpudata = convnet.CPUData(data)
        cpudata.thisown = 0

        if self.check_grads:
            worker = convnet.GradCheckWorker(self.model, cpudata)
            worker.thisown = 0
            self.model.getWorkerQueue().enqueue(worker)
            res = self.model.getResultQueue().dequeue()
            assert res.getResultType() == convnet.WorkResult.BATCH_DONE
        elif not train and self.multiview_test:
            worker = convnet.MultiviewTestWorker(self.model, cpudata,
                                                 self.tran_data_provider.num_views, self.logreg_idx)
            worker.thisown = 0
            self.model.getWorkerQueue().enqueue(worker)
        else:
            worker = convnet.TrainingWorker(self.model, cpudata, not train)
            worker.thisown = 0
            self.model.getWorkerQueue().enqueue(worker)
        
    def print_iteration(self):
        print "%d.%d..." % (self.epoch, self.batchnum),
        
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
        
    def print_costs(self, cost_outputs):
        costs, num_cases = cost_outputs[0], cost_outputs[1]
        for errname in costs.keys():
            costs[errname] = [(v/num_cases) for v in costs[errname]]
            print "%s: " % errname,
            print ", ".join("%6f" % v for v in costs[errname]),
            if sum(m.isnan(v) for v in costs[errname]) > 0 or sum(m.isinf(v) for v in costs[errname]):
                print "^ got nan or inf!"
                sys.exit(1)
        
    def print_train_results(self):
        #self.print_weight_summary()
        #print '=' * 40
        self.print_costs(self.train_outputs[-1])
        
    def print_test_status(self):
        pass
        
    def print_test_results(self):
        print ""
        print "======================Test output======================"
        self.print_costs(self.test_outputs[-1])
        print ""
        self.print_weight_summary()

    def print_weight_summary(self):
        print "-------------------------------------------------------", 
        for i,l in enumerate(self.layers): # This is kind of hacky but will do for now.
            if 'weights' in l:
                if type(l['weights']) == n.ndarray:
                    print "%sLayer '%s' weights: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
                elif type(l['weights']) == list:
                    print ""
                    print NL.join("Layer '%s' weights[%d]: %e [%e]" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
                print "%sLayer '%s' biases: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc']))),
        print ""

        
    def conditional_save(self):
        self.save_state()
        print "-------------------------------------------------------"
        print "Saved checkpoint to %s" % os.path.join(self.save_path, self.save_file)
        print "=======================================================",
        
    def aggregate_test_outputs(self, test_outputs):
        num_cases = sum(t[1] for t in test_outputs)
        for i in xrange(1 ,len(test_outputs)):
            for k,v in test_outputs[i][0].items():
                for j in xrange(len(v)):
                    test_outputs[0][0][k][j] += test_outputs[i][0][k][j]
        return (test_outputs[0][0], num_cases)
    
    @classmethod
    def get_options_parser(cls):
        op = OptionsParser()
        op.add_option("f", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCLUDE_ALL)
        op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
        op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
        op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
        op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
        op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
        op.add_option("data-path", "data_path", StringOptionParser, "Data path")
        op.add_option("save-path", "save_path", StringOptionParser, "Save path")
        op.add_option("max-filesize", "max_filesize_mb", IntegerOptionParser, "Maximum save file size (MB)", default=5000)
        op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
        op.add_option("num-gpus", "num_gpus", IntegerOptionParser, "Number of GPUs", default=1)
        op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
        op.add_option("zip-save", "zip_save", BooleanOptionParser, "Compress checkpoints?", default=0)
        op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
        op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override", default=OptionExpression("[-1] * num_gpus"))
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=True)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path','train_batch_range','test_batch_range'])
        op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0, requires=['logreg_name'])
        op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=4, set_once=True)
        op.add_option("logreg-name", "logreg_name", StringOptionParser, "Cropped DP: logreg layer name (for --multiview-test)", default="")
        op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
        op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
        op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)
                
        op.delete_option('max_test_err')
        op.options["max_filesize_mb"].default = 0
        op.options["testing_freq"].default = 50
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None
        
        DataProvider.register_data_provider('imagenet', 'ImageNet', ImageNetDataProvider)
        DataProvider.register_data_provider('cifar', 'CIFAR', CIFARDataProvider)
        DataProvider.register_data_provider('dummy-cn-n', 'Dummy ConvNet', DummyConvNetDataProvider)
        DataProvider.register_data_provider('cifar-cropped', 'Cropped CIFAR', CroppedCIFARDataProvider)
        
        return op
    
if __name__ == "__main__":
    #nr.seed(5)
    op = ConvNetRunner.get_options_parser()

    op, load_dic = ConvNetRunner.parse_options(op)
    model = ConvNetRunner(op, load_dic)
    model.start()


class ModelStateException(Exception):
    pass

