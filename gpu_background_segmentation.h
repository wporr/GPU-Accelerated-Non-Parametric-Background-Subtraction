#ifndef BACKSEG_H
#define BACKSEG_H
#include "imgproc.hpp"
#include "highgui.hpp"
#include "gpu_histogram.h"
#include "gpu_histogram.cpp"
#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>
#define NUM_STREAMS 2

using namespace cv;
using namespace std;

class background_segmentation {

private:
	 //Video file path and write path
	string vid_path;
	string write_path;
	 //Info about video feed
	int rows;
	int cols;
	int channels;
	int total_p;
	 //Counter variables
	int frame_num;
	 //Array of pixel histograms in the device
	Hist* d_histarr;
	 //Vector of all frames in the video
	vector<Mat> matArr;
	 //Number of cpu threads used
	int threads;
	 //Probability threashold
	double threashold = .006;
	 //Histogram update interval in terms of frames
	int update_interval = 8;
	 //Pointers to allocated page-locked memory for the binary and
	 // original frames
	uchar* framePtr[NUM_STREAMS];
	uchar* binPtr[NUM_STREAMS];
	 //Pointers to memory allocated on the device (GPU)
	uchar* d_binPtr;
	uchar* d_framePtr[NUM_STREAMS];
	 //Size variables
	size_t binSize;
	size_t frameSize;
	 //Cuda streams
	cudaStream_t stream[NUM_STREAMS];


public:
	background_segmentation(string, string, int);
	void allocate_mem();
	int get_frame_num();
	void run_video();
	string find_file_name(int, char = 'r');
	bool process_frame(Mat, int);
	void display_and_write(Mat, Mat, int);
	void check_for_errors();
};

#endif
