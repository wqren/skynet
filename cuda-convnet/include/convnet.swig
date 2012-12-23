%module "convnet"

#define __device__
#define MTYPE float

// A bunch of existing classes expect to take Python lists or
// dictionaries as inputs directly - let them
%typemap(in) PyListObject* { $1 = (PyListObject*)$input; }
%typemap(in) PyDictObject* { $1 = (PyDictObject*)$input; }

// Mapping from Python - C++ Matrix types
%typemap(in) std::vector<Matrix*> & { $1 = getMatrixV((PyObject*)$input); }


%typemap(out) std::map<std::string,std::vector<double>*>& {
  $result = PyDict_New();
  for (CostMap::const_iterator it = (*$1).begin(); it != (*$1).end(); ++it) {
    PyObject* v = PyList_New(0);
    for (vector<double>::const_iterator iv = it->second->begin(); iv != it->second->end(); ++iv) {
        PyObject* f = PyFloat_FromDouble(*iv);
        PyList_Append(v, f);
    }
    PyDict_SetItemString($result, it->first.c_str(), v);
  }
}

%exception {
  try {
    $action 
  } catch (std::string& s) {
	PyErr_SetString(PyExc_Exception, s.c_str());
	return NULL;
  } catch (char const*& s) {
	PyErr_SetString(PyExc_Exception, s);
	return NULL;
  }
}

%{

#include "common/matrix.h"
#include "common/queue.h"
#include "common/thread.h"
#include "common/logging.h"
#include <map>
#include <vector>

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

%include <std_map.i>
%include <std_list.i>
%include <std_vector.i>

%template(GPUData) Data<NVMatrix>;
%template(CPUData) Data<Matrix>;
%template(WorkerQueue) Queue<Worker*>;
%template(ResultQueue) Queue<WorkResult*>;
