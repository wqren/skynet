CXXOPT = -O3 -ggdb2
CXXINCLUDE = -I../
CXX = g++ ${CXXOPT} ${CXXINCLUDE}

libutil.a: logging.o common.o string.o
	ar rcs -o $@ $^

clean:
	rm *.o libutil.a
