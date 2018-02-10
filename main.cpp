#include "imgproc.hpp"
#include "highgui.hpp"
#include <iostream>
#include <vector>
#include "MyHistClass.h"
#include "MyHistClass.cpp"
#include <time.h>
#include <omp.h>
#include <fstream>
#include <ctime>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define NUM_THREADS 12
#define NUM_STREAMS 2
#define NUM_CHANNELS 4

using namespace cv;
using namespace std;

void print_gpu_stats();

int main(int argc, char* argv[]){

	print_gpu_stats();
}

void print_gpu_stats(){
//Print properties of the GPU
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	for (int i = 0; i < nDevices; i++){
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop,i);

		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf(" Support for concurrent memory copy and kernel execution: %d\n",
				prop.asyncEngineCount);
		printf(" Support for concurrent kernels: %d\n", prop.concurrentKernels);

		size_t memSize, free;
		cudaMemGetInfo(&free, &memSize);
		printf("Free memory %u\n", free);
	}
}
