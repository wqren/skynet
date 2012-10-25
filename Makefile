all:
	cd util && make -j
	cd cuda-convnet && ./build.sh verbose=1 -j
	cd zmq-play && make -j
	cd mpi-play && make -j
	
clean:
	cd util && make clean
	cd cuda-convnet && ./build.sh clean
	cd zmq-play && make clean
	cd mpi-play && make clean 
