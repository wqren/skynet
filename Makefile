all:
	cd cuda-convnet && ./build.sh verbose=1 -j2
	
clean:
	cd cuda-convnet && ./build.sh clean
