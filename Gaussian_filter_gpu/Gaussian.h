#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>

#include "opencv2/opencv.hpp"

#define ITER 16
#define KERNEL_LENGTH 10

using namespace cv;
using namespace std;
using namespace chrono;

extern "C" void blurring_gaussian(Mat src, float sigma, int ksize);
extern "C" void mat_to_vector(vector<float> &out, Mat in);
extern "C" void mat_to_vector2(vector<float> &out, Mat in);
extern "C" void filter_check(vector<float> filter);
extern "C" void inititalizedDataOne(vector<float>& container);
extern "C" void inititalizedData(vector<float>& container);
extern "C" void valueCheck2d(vector<float>& valueCheckInput, int input_h, int input_w);
extern "C" void array_to_mat(Mat out, vector<float> in);
extern "C" void int_array_to_mat(Mat out, vector<int> in);

extern "C" void setConvolutionKernel(float *h_Kernel);
extern "C" void convolution(vector<float>& convOutput, vector<float>& convInput, vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c);
extern "C" void gaussianFilter1d(vector <float> &output, float sigma, int k_radius, int odd);
extern "C" void gaussianFilter2d(vector <float> &output, float sigma, int k_radius, int odd);
extern "C" void reflectivePadding(vector<float>& output, vector<float>& input, int h, int w, int k_radius, int odd);
extern "C" void valueCheck2d_6(vector<float>& valueCheckInput, int input_h, int input_w);

extern "C" void conv1d_vertical(float* d_output,  float* d_data,  int H, int W, int k_radius, int odd);
extern "C" void vertical_reflect_padding(float* d_output,  float* d_data,  int H, int W, int k_radius, int odd);
extern "C" void conv1d_horizontal(float* d_output,  float* d_data,  int H, int W, int k_radius, int odd);
extern "C" void horizontal_reflect_padding(float* d_output, float* d_data,  int H, int W, int k_radius, int odd);

