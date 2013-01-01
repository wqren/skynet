#/usr/bin/env python

import StringIO
import os, sys
import numpy as N
import numpy.distutils.system_info as S

numpy = os.path.dirname(N.__file__) + '/core/include/numpy'
python = os.popen("python-config --includes").read()

python = [p[2:] for p in python.split(" ")]
mpi_include = os.popen("mpicxx -showme:compile").read()
mpi_link = os.popen("mpicxx -showme:link").read()

print "export ATLAS_LIB_PATH=/usr/lib/openblas-base"
print "export NUMPY_INCLUDE_PATH=%s" % numpy
print "export PYTHON_INCLUDE_PATH=%s" % python[0]
