CC=g++
NVCC=nvcc
CFLAGS=-Wall -Wextra -std=c++11
LIBS=`pkg-config opencv4 --cflags --libs`

run:
	$(CC) $(CFLAGS) compression.cpp -o compression $(LIBS)
	$(NVCC)  test.cu -o test $(LIBS)
	$(NVCC)  parallel_compression.cu -o parallel_compression $(LIBS)

clean:
	rm -f compression parallel_compression
