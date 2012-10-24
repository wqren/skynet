LIBS := -L../util/ -lboost_program_options -lutil -lrt

mpi-perf: mpi-perf.cc Makefile
	mpic++ -I.. -O3 -fno-omit-frame-pointer -ggdb1 mpi-perf.cc ${LIBS} -o mpi-perf

clean:
	rm -f mpi-perf a.out
