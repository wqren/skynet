all:
	cd cuda-convnet && ./build.sh dbg=1 verbose=1 -j2
	
clean:
	cd cuda-convnet && ./build.sh clean
