all:
	cd cuda-convnet && ./build.sh verbose=1 -j
	cd util && make -j
	cd zmq-play && make -j
	cd mpi-play && make -j
	
clean:
	cd cuda-convnet && ./build.sh clean
	cd util && make clean
	cd zmq-play && make clean
	cd mpi-play && make clean 
