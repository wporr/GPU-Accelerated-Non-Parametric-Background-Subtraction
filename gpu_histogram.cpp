#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "gpu_histogram.h"
#include "imgproc.hpp"
#include "highgui.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

__device__ __host__ Hist::Hist(int nobins){	// dont call this constructior with any arguments, it will throw.
	bins = nobins;
	timesupdated = 0;

	for (int i = 0; i < bins; i++)
		hist[i] = 0;
	binfactor = (256 / bins);
}


__device__ __host__ void Hist::updateHist(uchar& intensity){
	hist[(int)(intensity / binfactor)] += 1;
	timesupdated++;
}



__device__ __host__ float Hist::getBinVal(uchar& intensity){
	//If there are no stored values, we cannot return anything
	if (timesupdated == 0)
		return NULL;
	//Normalize values
	return (hist[(int)(intensity / binfactor)] / timesupdated);
}

__device__ __host__ void Hist::operator=(Hist& obj) {
	timesupdated = obj.timesupdated;
	for (int i = 0; i < bins; i++)
		hist[i] = obj.hist[i];

}
