#include "gpu_background_segmentation.h"
#include "imgproc.hpp"
#include "highgui.hpp"
#include <iostream>
#include <vector>
#include <omp.h>
#define NUM_STREAMS 2

using namespace std;
using namespace cv;

background_segmentation::background_segmentation(string vid_path_i,
						 						 string write_path_i,
                                                 int threads_i){
	vid_path = vid_path_i;
	write_path = write_path_i;
	string first_name = find_file_name(0);
	Mat frame = imread(vid_path + first_name);

	if (frame.empty()){
		cout << "Video did not open";
		return;
	}

	frame_num = 0;
	rows = frame.rows;
	cols = frame.cols;
	channels = frame.channels();
	total_p = rows * cols;

	//Dynamically allocate the number of histograms needed
	histarr = new Hist[total_p * channels];
	histarr[0].initialize_table();	//initialize static lookup table

	//Set number of threads for OpenMP
	threads = threads_i;
	omp_set_num_threads(threads);

	 //Place all the frames of the video sequence in a vector array
	totalFrames = findVideo(matArr, readfile);

	 //Create NUM_STREAMS
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamCreate(&stream[i]);
}

void background_segmentation::allocate_mem(){
	 //Allocated MyHist array to hold the histograms for each pixel
	Hist* histarr = new Hist[total_p * channels];

	 //Allocate memory in device and transfer histarr
	size_t size = rows * cols * channels * sizeof(Hist);
	cout << "Space needed in gpu for histograms: " << size;

	cudaError_t g = cudaMalloc(&d_histarr, size);
	cudaError_t r = cudaMemcpy(d_histarr, histarr, size, cudaMemcpyHostToDevice);

	 //Clear histarr mem because we wont be using it
	delete histarr;

	 // allocate memory slots for the binary and original frame in the device
	binSize = sizeof(uchar) * rows * cols;
	frameSize = sizeof(uchar) * rows * cols * channels;

	cudaMalloc(&d_binPtr, binSize * NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++)	{
		cudaMallocHost(&framePtr[i], frameSize);
		cudaMalloc(&d_framePtr[i], frameSize);
		cudaMallocHost(&binPtr[i], binSize);
		cudaError_t a = cudaGetLastError();

		if (a != cudaSuccess) {
			cout << "Cuda error: " << a << endl;
		}
	}
}


bool background_segmentation::findVideo(int n, int curr_frame_num, Mat* matArr) {

	 //Identify next frame and store it in mat obj
	for (int i=0; i < n; i++) {
		string file_name = find_file_name(curr_frame_num);
		Mat frame = imread(vid_path + file_name);

		if (curr_frame_num - i == 0)
			matArr.push_back(frame);
		else
			matArr[i] = frame;

		if (matArr[i].empty()) {
			cout << "Video feed has ended on frame: "
				 << curr_frame_num + i << endl;

			return true;
		}
		 //Convert to BGRR frames for warp optimization
		else {
			Mat source = matArr[i];
			Mat newSrc(source.size(), CV_MAKE_TYPE(source.type(), 4));
			int from_to[] = { 0,0, 1,1, 2,2, 2,3 };
			mixChannels(&source, 1, &newSrc, 1, from_to, 4);
			matArr[i] = newSrc;
		}
	}
	return false;
}

int background_segmentation::get_frame_num(){
	return frame_num;
}

void background_segmentation::run_video(){

	bool end = true;
	int frames_to_process = NUM_STREAMS * threads;
	Mat* mats = new Mat[frames_to_process];

	do {
		#pragma omp parallel
		{
			Mat* curr_mats[NUM_STREAMS];
			for (int i=0; i < NUM_STREAMS; i++)
				curr_mats[i] = mats[i + (omp_get_thread_num*NUM_STREAMS)]

			int curr_frame_num = frame_num + omp_get_thread_num();
			end = findVideo(NUM_STREAMS, curr_frame_num, mats)
			process_frame(curr_mats, curr_frame_num);
			write(curr_mats);
		}
		frame_num += frames_to_process;
	} while (end == false);

	return;
}

//This function is specifically for finding the file name for a video frame
// from the Changedetection.net datasets
string background_segmentation::find_file_name(int curr_frame_num,
					       					   char mode){

	string file_name = (mode == 'r') ? "in" : "out";
	string path = (mode == 'r') ? vid_path : write_path;
	int o_num = 6 - to_string(curr_frame_num + 1).length();

	if (path.find_first_of('/') == string::npos)
		file_name = (path.back() == '\\') ? file_name : "\\" + file_name;
	else
		file_name = (path.back() == '/') ? file_name : "/" + file_name;

	for (int i = 0; i < o_num; i++)
		file_name += '0';

	file_name += to_string(curr_frame_num + 1);
	file_name = (mode == 'r') ? file_name + ".jpg": file_name;

	return file_name;
}

//Main function that processes the input frame
void background_segmentation::process_frame(Mat* curr_mats
											int curr_frame_num) {
	clock_t start = clock();
	Mat binary[NUM_STREAMS];
	bool update = false; // update flag for kernel
	int fornum = 1; // counter for light detection function

	if (curr_frame_num % 8 == 0)
		update = true;
	 //Assign the page-locked array pointers to the mat object pointers
	for (int i = 0; i < NUM_STREAMS; i++) {
		framePtr[i] = matArr[frameno + i].ptr<uchar>(0);

	 //Only copy main frame because the binary is blank
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaMemcpyAsync(d_framePtr[i], framePtr[i], frameSize, cudaMemcpyHostToDevice, stream[i]);

	 //Check for errors before executing
	check_for_errors();

	 //Execute kernel
	for (int i = 0; i < NUM_STREAMS; i++)
		processFrameKernel<<<dimGrid, dimBlock, 0, stream[i]>>>(d_histarr, d_framePtr[i], d_binPtr + i * binSize, cols, update);

	  //Check again for errors
	 check_for_errors();
	 //Copy the binary image from device memory to host memory
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaMemcpyAsync(binPtr[i], d_binPtr + i * binSize, binSize, cudaMemcpyDeviceToHost, stream[i]);

	duration += (clock() - start);
}

void background_segmentation::display_and_write(Mat* curr_mats,
                                                int curr_frame_num){
	for (int i=0; i < NUM_STREAMS; i++) {
		string file_name = find_file_name(curr_frame_num, 'w');
		imwrite(write_path + file_name + ".jpg", curr_mats[i]);
	}
}

void background_segmentation::light_change(Mat& frame)
{
	uchar* frameptr = frame.ptr<uchar>(0);
	for (int count = 0; count < (frame.rows * frame.cols * 3); count++)
	{
		histarr[count].clear_hist();
		histarr[count].update_hist(frameptr[count]);
	}
	return;
}

void background_segmentation::check_for_errors(){
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
		cout << "cuda error: " << e;
}


__global__ void processFrameKernel(MyHist* histArr,
								   uchar* framePtr,
								   uchar* binPtr,
								   int width,
								   bool update){

	// find our location in the grid
	//int fornum = 0;
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int threadRow = threadIdx.y;
	int threadCol = threadIdx.x;
	// create pointers to the block we're on in the various matricies
	uchar* fSub = &framePtr[width * blockDim.y * blockRow * NUM_CHANNELS + blockDim.x * blockCol * NUM_CHANNELS];
	MyHist* hSub = &histArr[width * blockDim.y * blockRow * 3 + blockDim.x * blockCol * 3];
	uchar* bSub = &binPtr[width * blockDim.y * blockRow + blockDim.x * blockCol];
	// load frame intensities to local mem
	uchar bIntens = fSub[threadRow * width * NUM_CHANNELS + threadCol * NUM_CHANNELS];
	uchar gIntens = fSub[threadRow * width * NUM_CHANNELS + threadCol * NUM_CHANNELS + 1];
	uchar rIntens = fSub[threadRow * width * NUM_CHANNELS + threadCol * NUM_CHANNELS + 2];
	// create shared memory and assign pointer to location for local thread
	// extract information and determine probability
	float probability = hSub[threadRow * width * 3 + threadCol * 3].getBinVal(bIntens) * hSub[threadRow * width * 3 + threadCol * 3 + 1].getBinVal(gIntens) *
		hSub[threadRow * width * 3 + threadCol * 3 + 2].getBinVal(rIntens);
	// set binary location to 255 if it is determined as foreground
	// or zero if we determine it to be background
	if (probability < .03)
		bSub[threadRow * width + threadCol] = 255;
	else if (probability >= .03 && probability <= 1)
		bSub[threadRow * width + threadCol] = 0;
	// update the histogram if we are on the correct frame
	if (update) {
		for (int i = 0; i < 3; i++) {
			uchar intensity;
			switch (i) {
				case 0: { intensity = bIntens;
				   		break; }
				case 1: { intensity = gIntens;
						break; }
				case 2: { intensity = rIntens;
						break; }
			}
			hSub[threadRow * width * 3 + threadCol * 3 + i].updateHist(intensity);
		}
	}
	// synch threads to prevent read/write issues
	__syncthreads();
	// process through median filter for each block independently
	if ((blockCol == 0 && threadCol > 0) || (blockCol > 0  && blockCol < (gridDim.x - 1)) ||
	 				(blockCol == (gridDim.x - 1) && threadCol < (blockDim.x - 1)) ) {
		if ((blockRow == 0 && threadRow > 0) || (blockRow > 0 && blockRow < (gridDim.y - 1)) ||
						(blockRow == (gridDim.y - 1) && threadRow < (blockDim.y - 1)) ) {
			if (bSub[threadRow * width + threadCol] == 255) {
				int totalVal = 0, avgVal = 0;
				for (int count = (threadRow - 1); count < (threadRow + 2); count++) {
					for (int i = (threadCol - 1); i < (threadCol + 2); i++)
						totalVal += bSub[count * width + i];
				}
				// find average value and assign black or white based on the result
				avgVal = totalVal / 9;
				if (avgVal < 128)
					bSub[threadRow * width + threadCol] = 0;
			}
		}
	}
}
