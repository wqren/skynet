CXXOPT = -O3 -ggdb2 -fPIC
CXXINCLUDE = -I../
CXX = g++ ${CXXOPT} ${CXXINCLUDE}

libutil.a: logging.o common.o string.o
	ar rcs -o $@ $^

clean:
	rm *.o libutil.a
common.o: common.cc common.h
logging.o: logging.cc logging.h string.h common.h
string.o: string.cc string.h
