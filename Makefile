CC=g++
CFLAGS=-Wall -Wextra -std=c++11
LIBS=`pkg-config opencv4 --cflags --libs`

compression: compression.cpp
	$(CC) $(CFLAGS) compression.cpp -o compression $(LIBS)

.PHONY: clean
clean:
	rm -f compression
