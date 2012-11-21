#/usr/bin/env python

import StringIO
import os, sys
import numpy.distutils.system_info as S

atlas = S.get_info("atlas_blas")
numpy = S.get_info("numpy")
python = os.popen("python-config --includes").read()
python = [p[2:] for p in python.split(" ")]
mpi_include = os.popen("mpicxx -showme:compile").read()
mpi_link = os.popen("mpicxx -showme:link").read()

print "export ATLAS_LIB_PATH=%s" % atlas["library_dirs"][0]
print "export NUMPY_INCLUDE_PATH=%s/numpy" % numpy["include_dirs"][0]
print "export PYTHON_INCLUDE_PATH=%s" % python[0]
