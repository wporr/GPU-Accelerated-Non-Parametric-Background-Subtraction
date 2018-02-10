OPENCV=`pkg-config opencv --cflags --libs`
CFLAGS=-Wall -Wextra --pedantic -g1 -O3 -std=c++0x
CUDA='/opt/cuda/include'
all:Main.out

Main.out:
	g++ -o Main.out main.cpp $(OPENCV) $(CFLAGS) -fopenmp -I $(CUDA)

clean:
	rm Main.out
