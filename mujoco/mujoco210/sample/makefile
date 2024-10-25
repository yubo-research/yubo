COMMON=-O2 -I../include -L../bin -std=c++11 -stdlib=libc++ -mavx -pthread

all:
	clang++ $(COMMON) testxml.cpp    -lmujoco200nogl      -o ../bin/testxml
	clang++ $(COMMON) testspeed.cpp  -lmujoco200nogl      -o ../bin/testspeed
	clang++ $(COMMON) compile.cpp    -lmujoco200nogl      -o ../bin/compile
	clang++ $(COMMON) derivative.cpp -lmujoco200nogl      -o ../bin/derivative
	clang++ $(COMMON) basic.cpp      -lmujoco200 -lglfw.3 -o ../bin/basic
	clang++ $(COMMON) record.cpp     -lmujoco200 -lglfw.3 -o ../bin/record
	clang -c -O2 -mavx -I../include ../include/uitools.c
	clang++ $(COMMON) uitools.o simulate.cpp -lmujoco200 -lglfw.3 -o ../bin/simulate
	rm *.o