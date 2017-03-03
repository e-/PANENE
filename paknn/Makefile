CC=g++
CFLAGS=-c -Wall -fopenmp -std=c++11 -O2 -I/usr/include/hdf5/serial 
LDFLAGS=-fopenmp -L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib -L/usr/lib/x86_64-linux-gnu -lhdf5 -llz4 lib/lz4.o lib/lz4hc.o
SOURCES=knn_table.cpp paknn.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE) 
	./main

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o main
