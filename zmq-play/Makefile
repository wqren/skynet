CXXOPT = -O3
CXXINCLUDE = -I.
CXX = g++ ${CXXOPT} ${CXXINCLUDE}

all: server client
clean:
	rm -f *.o util/*.o server client

server:  server.o util/common.o util/logging.o util/string.o
	${CXX} $^ -lzmq -o $@

client:  client.o util/common.o util/logging.o util/string.o
	${CXX} $^ -lzmq -o $@
