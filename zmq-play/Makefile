CXXOPT = -O3 -ggdb2
CXXINCLUDE = -I../
LIBS = -L../util/ -lzmq -lboost_program_options -lutil
CXX = g++ ${CXXOPT} ${CXXINCLUDE}

all: server client
clean:
	rm -f *.o util/*.o server client

server:  server.o 
	${CXX} $^ ${LIBS} -o $@

client:  client.o 
	${CXX} $^ ${LIBS} -o $@
