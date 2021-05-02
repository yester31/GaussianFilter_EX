// reflective_padding.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <math.h>

using namespace std; 

void convolution(vector<float>& convOutput, vector<float>& convInput, vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
	int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
	int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
	//Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
	//cout << "===== Convolution ===== \n";

	int temp1i = input_h * input_w * input_c;
	int temp1o = outputHeightSize * outputWidthSize * ouput_c;
	int temp1k = kernelSize * kernelSize * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < ouput_c; k_idx++)
		{
			int temp2k = k_idx * temp1k;
			int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
					int temp4o = rowStride * outputWidthSize + temp3o;
					for (int colStride = 0; colStride < outputWidthSize; colStride++) {

						float sum = 0;
						int g_idx_o = colStride + temp4o;
						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							int temp4i = x * input_w + temp3i;
							int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								int ⁠g_idx_i = y + temp4i;
								int g_idx_k = (y - colStride * stride) + temp4k;
								sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
							}
						}
						convOutput[g_idx_o] += sum;
					}
				}
			}
		}
	}
}

void valueCheck2d(vector<float>& valueCheckInput, int input_h, int input_w)
{

	for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++) {
		int temp4 = ⁠h_idx * input_w; cout << "  ";
		for (int w_idx = 0; w_idx < input_w; w_idx++)
		{
			int g_idx = w_idx + temp4;
			cout.setf(ios::fixed);
			cout.precision(6);
			cout << setw(9) << valueCheckInput[g_idx] << " ";

		}cout << endl;
	}cout << endl;

}

void inititalizedData(vector<float>& container)
{
	int count = 0;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
		count++;
	}
}

void reflectivePadding(vector<float>& output, vector<float>& input, int h, int w, int k_radius, int odd )
{

	vector<float> data_p(h  * ((w + k_radius * 2) - odd));

	for (int ⁠h_idx = 0; ⁠h_idx < h; ⁠h_idx++) {

		int offset = ⁠h_idx * ((w + k_radius * 2) - odd);

		for (int w_idx = 0; w_idx < (w + k_radius * 2) - odd; w_idx++) {

			int g_idx = w_idx + offset;

			if (w_idx < k_radius - odd) {
				int i_idx = k_radius - odd - w_idx + ⁠h_idx * w;
				data_p[g_idx] = input[i_idx];
			}
			else if (w_idx >= w + k_radius - odd) {
				int i_idx = 2 * w - w_idx + k_radius - odd + ⁠h_idx * w - 2;
				data_p[g_idx] = input[i_idx];

			}
			else {

				int i_idx = w_idx - (k_radius - odd) + ⁠h_idx * w;
				data_p[g_idx] = input[i_idx];
			}
		}
	}




	for (int ⁠h_idx = 0; ⁠h_idx < (h + k_radius * 2) - odd; ⁠h_idx++) {

		int offset = ⁠h_idx * (w + k_radius * 2 - odd);

		for (int w_idx = 0; w_idx < w + k_radius * 2 - odd; w_idx++) {

			int g_idx = w_idx + offset;

			if (⁠h_idx < k_radius - odd) {

				int i_idx = ((k_radius - odd) - ⁠h_idx) * (w + k_radius * 2 - odd) + w_idx;

				output[g_idx] = data_p[i_idx];

			}
			else if (⁠h_idx > h + k_radius - (1 + odd)) {

				int i_idx = (h * 2 + (k_radius - odd) - (⁠h_idx + 2))* (w + k_radius * 2 - odd) + w_idx;

				output[g_idx] = data_p[i_idx];

			}
			else {

				int i_idx = g_idx - ((k_radius - odd) * ((w + k_radius * 2) - odd));

				output[g_idx] = data_p[i_idx];

			}
		}
	}
}

void gaussianFilter2d(vector <float> &output, float sigma, int k_radius, int odd) {
	float pi = 3.14159265358979f;
	float summation = 0;
	for (int y = -k_radius; y <= k_radius - odd; y++) {
		int y_idx = y + k_radius;
		int offset = (y_idx)* (k_radius * 2 + 1 - odd);
		for (int x = -k_radius; x <= k_radius - odd; x++) {
			int x_idx = x + k_radius;
			int g_idx = offset + x_idx;
			output[g_idx] = (1. / (2. * pi * pow(sigma, 2))) * exp(-((pow(x, 2) + pow(y, 2)) / (2. * pow(sigma, 2))));
			summation += output[g_idx];
		}
	}

	for (int y = -k_radius; y <= k_radius - odd; y++) {
		int y_idx = y + k_radius;
		int offset = (y_idx)* (k_radius * 2 + 1 - odd);
		for (int x = -k_radius; x <= k_radius - odd; x++) {
			int x_idx = x + k_radius;
			int g_idx = offset + x_idx;
			output[g_idx] /= summation ;
		}
	}
}

void gaussianFilter1d(vector <float> &output, float sigma, int k_radius, int odd) {
	float pi = 3.14159265358979f;
	
	for (int y = -k_radius; y <= k_radius - odd; y++) {
		int y_idx = y + k_radius;
		int offset = (y_idx)* (k_radius * 2 + 1 - odd);
		float sum = 0;
		for (int x = -k_radius; x <= k_radius - odd; x++) {
			int x_idx = x + k_radius;
			int g_idx = offset + x_idx;

			sum += (1. / (2. * pi * pow(sigma, 2))) * exp(-((pow(x, 2) + pow(y, 2)) / (2. * pow(sigma, 2))));

		}
		output[y_idx] = sum;
	}
}

void filter_check(vector<float> filter) {
	cout << "filter size(ksize) :: " << filter.size() << endl;
	for (int i = 0; i < filter.size(); i++) {
		cout << filter[i] << endl;
	}
}

void inititalizedDataOne(vector<float>& container)
{
	int count = 1;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
	}
}

int main()
{
	int h = 10;
	int w = 10;

	int ksize = 7;
	int k_radius = ksize / 2;

	float sigma = 1.0;
	int odd = 0;

	if (ksize % 2 == 0) {
		 odd = 1;
	}

	//cpu 함수 출력 벡터
	vector<float> output(h * w);

	vector<float> data(h * w);

	inititalizedData(data);
	//inititalizedDataOne(data);
	vector<float> data_pv(((h + k_radius * 2) - odd) * ((w + k_radius * 2) - odd));

	// reflect padding
	reflectivePadding(data_pv, data, h, w, k_radius, odd);

	// 계산 값 확인
	valueCheck2d(data_pv, h + k_radius * 2 - odd, w + k_radius * 2 - odd);

	vector <float> GaussianFilter(ksize*ksize);

	// 2d 가우시안 필터 생성
	//gaussianFilter2d(GaussianFilter, sigma, k_radius, odd);
	//inititalizedDataOne(GaussianFilter);
	inititalizedData(GaussianFilter);

	// 계산 값 확인
	valueCheck2d(GaussianFilter, ksize, ksize);

	// 1d 가우시안 필터 생성
	//vector <float> GaussianFilter1d(ksize);
	//gaussianFilter1d(GaussianFilter1d, sigma, k_radius, odd);
	//filter_check(GaussianFilter1d);

	// filtering, 컨볼류션
	convolution(output, data_pv, GaussianFilter, ksize, 1, 1, 1, h + k_radius * 2 - odd, w + k_radius * 2 - odd, 1);

	// 계산 값 확인
	valueCheck2d(output, h, w);


}


