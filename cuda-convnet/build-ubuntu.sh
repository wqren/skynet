#!/bin/sh

# Fill in these environment variables.
# I have tested this code with CUDA 4.0, 4.1, and 4.2. 
# Only use Fermi-generation cards. Older cards won't work.

# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/usr/local/cuda-4.2

# CUDA SDK installation directory.
export CUDA_SDK_PATH=/usr/local/gpu-sdk

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/usr/include/python2.7

# Numpy include directory. This should contain the file arrayobject.h, among others.
#export NUMPY_INCLUDE_PATH=/usr/lib/pymodules/python2.7/numpy/core/include/numpy
export NUMPY_INCLUDE_PATH=/usr/share/pyshared/numpy/core/include/numpy/

# MPI include directories.  You should be able to get these from running mpicxx -showme:compile
export MPI_INCLUDE="-I/home/power/pkg/openmpi/include"
export MPI_LINK="-L/home/power/pkg/openmpi/lib -lmpi_cxx -lmpi -lrdmacm -libverbs -lrt -lnsl -lutil -lm -ldl -lm -lrt -lnsl -lutil -lm"

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/usr/lib/atlas-base

make $*

