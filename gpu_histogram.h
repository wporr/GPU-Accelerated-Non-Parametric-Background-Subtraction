#ifndef MYHIST_H
#define MYHIST_H
#include "imgproc.hpp"
#include "highgui.hpp"
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;


class Hist
{
private:
	float binfactor;
	// must change the number in the array below to the value
	// of the history variable manually, dynamic allocaiton is too slow
	// bins set to 6 for now, to change it, change this and
	// the constructor's default value
//		int lookuptable[256];
	float bins;
	float timesupdated;
	int hist[6];

public:
	__device__ __host__ Hist(int = 6);
	__device__ __host__ void updateHist(uchar&);
	__device__ __host__ float getBinVal(uchar&);
	__device__ __host__ void clearHist(){
		for (int i = 0; i < bins; i++)
			hist[i] = 0;
		timesupdated = 0;
	}
	__device__ __host__ void operator=(Hist&);
};

#endif
