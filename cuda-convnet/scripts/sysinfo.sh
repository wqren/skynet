#!/bin/bash

function test_result() {
  if [[ ! $? ]]; then
    rm -f /tmp/convnet-config
    exit 1
  fi
}

if [[ ! -f /tmp/convnet-config ]]; then
  mpicomp=$(mpicxx -showme:compile)
  test_result
  mpilink=$(mpicxx -showme:link)
  test_result

  echo export MPI_INCLUDE="'$mpicomp'" >> /tmp/convnet-config
  echo export MPI_LINK="'$mpilink'" >> /tmp/convnet-config
  python ./sysinfo.py | egrep "LIB|INCLUDE" >> /tmp/convnet-config
  test_result
  sed -i -e 's/-pthread//' /tmp/convnet-config
fi
