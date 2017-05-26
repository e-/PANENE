CC=g++
CFLAGS=-c -Wall -std=c++11 -I/usr/include/hdf5/serial -g -O2 -fopenmp
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib -L/usr/lib/x86_64-linux-gnu -lhdf5 -llz4 lib/lz4.o lib/lz4hc.o -fopenmp
HEADERS=
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(HEADERS) $(SOURCES) $(EXECUTABLE) 
	./main

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o *.gch main
	rm -f index/*.o index/*.gch util/*.gch table/*.o table/*.gch

