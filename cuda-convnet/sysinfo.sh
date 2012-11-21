#!/bin/bash

if [[ ! -f /tmp/convnet-config ]]; then
  rm -f /tmp/convnet-config
  echo export MPI_INCLUDE="'$(mpicxx -showme:compile)'" >> /tmp/convnet-config
  echo export MPI_LINK="'$(mpicxx -showme:link)'" >> /tmp/convnet-config
  python ./sysinfo.py | egrep "LIB|INCLUDE" >> /tmp/convnet-config

  sed -i -e 's/-pthread//' /tmp/convnet-config
fi
