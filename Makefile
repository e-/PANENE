CC=g++-6
CFLAGS=-c -Wall -std=c++11 -O2 -fopenmp
LDFLAGS=-I ../flann/src/cpp/flann -lflann -lhdf5 -fopenmp
SOURCES=main.cpp timer.cpp # hello.cpp factorial.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE) 
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o main
