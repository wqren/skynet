#!/usr/bin/env python

from pandas import *
from glob import glob

d = dict([(f, [float(x) for x in file(f).read().split(',')])
          for f in glob('out.*/mpi.1.0')])

df = DataFrame(d).median()
sizes = df.keys().map(lambda k: int(re.split('[./]', k)[1]))

print sizes * (1/df)

