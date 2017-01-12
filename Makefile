CC=g++
CFLAGS=-c -Wall -std=c++11 -O2
LDFLAGS=-I ../flann/src/cpp/flann -lflann -lhdf5
SOURCES=main.cpp # hello.cpp factorial.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE) 
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@
$(OBJECTS): $(SOURCES)
	$(CC) $< $(CFLAGS) -o $@

clean:
	rm *.o main
