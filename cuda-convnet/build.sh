#!/bin/bash

# Fill in these environment variables.
# I have tested this code with CUDA 4.0, 4.1, and 4.2. 
# Only use Fermi-generation cards. Older cards won't work.

# Try to determine various configurations automatically from numpy/python-config
function fail() {
  echo Failed to determine system info.
  exit 1
}

./scripts/sysinfo.sh || fail
source /tmp/convnet-config

echo Found ATLAS: ${ATLAS_LIB_PATH}
echo Found NUMPY: ${NUMPY_INCLUDE_PATH}
echo Found PYTHON_INCLUDE_PATH: ${PYTHON_INCLUDE_PATH}
echo Found MPI_INCLUDE: ${MPI_INCLUDE}
echo Found MPI_LINK: ${MPI_LINK}

#export ATLAS_LIB_PATH=...
#export NUMPY_INCLUDE_PATH=...
#export PYTHON_INCLUDE_PATH=...
#export MPI_INCLUDE=...
#export MPI_LINK=...

# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/home/power/pkg/cuda-5.0

# CUDA SDK installation directory.
export CUDA_SDK_PATH=/home/power/pkg/gpusdk

swig -I./include -c++ -Wall -python -O -threads -o src/_convnet.cu include/convnet.swig
mv src/convnet.py .

make $*

