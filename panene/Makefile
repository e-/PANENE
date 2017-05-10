CC=g++
CFLAGS=-c -Wall -std=c++11 -O2 -I/usr/include/hdf5/serial -g #-fopenmp
LDFLAGS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib -L/usr/lib/x86_64-linux-gnu -lhdf5 -llz4 lib/lz4.o lib/lz4hc.o #-fopenmp
HEADERS=knn_table.h knn_table_flann.h indices/kd_tree_index.h indices/progressive_kd_tree_index.h test/kdtree_test.h 
SOURCES=panene.cpp
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
	rm -f indices/*.o indices/*.gch util/*.gch

