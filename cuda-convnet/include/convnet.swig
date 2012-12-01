%module "convnet"

#define __device__
#define MTYPE float

%typemap(in) PyListObject* {
    $1 = (PyListObject*)$input;
}

%typemap(in) PyDictObject* {
    $1 = (PyDictObject*)$input;
}

%exception {
  try {
    $action 
  } catch (std::string& s) {
	PyErr_SetString(PyExc_Exception, s.c_str());
	return NULL;
  } catch (const char* s) {
	PyErr_SetString(PyExc_Exception, s);
	return NULL;
  }
}

%{

#include "common/matrix.h"
#include "common/queue.h"
#include "common/thread.h"
#include "common/logging.h"
#
#include "nvmatrix/nvmatrix.cuh"
#include "convnet.cuh"
#include "cost.cuh"
#include "data.cuh"
#include "layer.cuh"
#include "layer_kernels.cuh"
#include "neuron.cuh"
#include "util.cuh"
#include "weights.cuh"
#include "worker.cuh"


%}

%include "common/matrix.h"
%include "common/queue.h"
%include "common/thread.h"
%include "common/logging.h"
%include "nvmatrix/nvmatrix.cuh"
%include "convnet.cuh"
%include "cost.cuh"
%include "data.cuh"
%include "layer.cuh"
%include "layer_kernels.cuh"
%include "neuron.cuh"
%include "util.cuh"
%include "weights.cuh"
%include "worker.cuh"
