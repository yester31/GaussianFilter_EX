#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "Gaussian.h"


#if defined(NDEBUG)
#define CUDA_CHECK(x) (x) // release mode
#else //debug mode
#define CUDA_CHECK(x) do{\
		(x); \
cudaError_t e = cudaGetLastError();\
if(cudaSuccess != e){\
	printf("cuda failure \n *** %s ***\n at %s :: line_  %d\n",\
			cudaGetErrorString(e), \
			__FILE__, __LINE__); \
		exit(1);\
	}\
}while(0)
#endif


int main(int argc, char **argv)
{

	int test_H = 10;
	int test_W = 10;
	int test_k_radius = KERNEL_LENGTH / 2;
	int kernel_size = KERNEL_LENGTH;
	cout << "ksize :: " << KERNEL_LENGTH << endl;

	int odd = 0;
	if (KERNEL_LENGTH % 2 == 0) {
		kernel_size += 1;
		odd = 1;
	}

	// opencv 가우시안 필터 생성
	Mat GaussianFilter = getGaussianKernel(kernel_size, (float)1);

	// 필터를 담을 벡터 생성
	vector<float> filter(kernel_size);

	// mat 형식 필터를 벡터로 변경
	mat_to_vector(filter, GaussianFilter);




	// filter 값들을 모두 1로 설정
	//inititalizedDataOne(filter);

	// vector로 변경한 필터 값 체크 
	//filter_check(filter);

	float *h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));

	for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
	{
		h_Kernel[i] = filter[i];
		cout << filter[i] << endl;
	}

	setConvolutionKernel(h_Kernel);

	vector<float> test100(test_W * test_H);
	vector<float> test100_result(test_W * test_H);
	inititalizedData(test100);
	//inititalizedDataOne(test100);
	cout << "Original img" << endl;
	valueCheck2d(test100, test_H, test_W);

	float* d_data;	   // 이미지
	(int)cudaMalloc(&d_data, test100.size() * sizeof(float));
	(int)cudaMemcpy(d_data, test100.data(), test100.size() * sizeof(float), cudaMemcpyHostToDevice);


	vector<float> p_output((test_W + 2 * test_k_radius - odd) * test_H);
	float* d_p_output;
	(int)cudaMalloc(&d_p_output, p_output.size() * sizeof(float));

	vector<float> pv_output((test_H + 2 * test_k_radius - odd) * test_W);
	float* d_pv_output;
	(int)cudaMalloc(&d_pv_output, pv_output.size() * sizeof(float));

	float* d_output;	// 결과값
	(int)cudaMalloc(&d_output, test100.size() * sizeof(float));

	float* df_output;	// 결과값
	(int)cudaMalloc(&df_output, test100.size() * sizeof(float));

	uint64_t total_time = 0;
	for (int iIdx = 0; iIdx < ITER; iIdx++) {
		uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();


		horizontal_reflect_padding(d_p_output, d_data, test_H, test_W, test_k_radius, odd);

		conv1d_horizontal(d_output, d_p_output, test_H, test_W, test_k_radius, odd);

		vertical_reflect_padding(d_pv_output, d_output, test_H, test_W, test_k_radius, odd);

		conv1d_vertical(df_output, d_pv_output, test_H, test_W, test_k_radius, odd);

		total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
	}

	printf("GPU GaussianBlur dur_time=%6.3f[msec] \n", total_time / 1000.f / ITER);

	(int)cudaMemcpy(p_output.data(), d_p_output, p_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	(int)cudaMemcpy(test100.data(), d_output, test100.size() * sizeof(float), cudaMemcpyDeviceToHost);
	(int)cudaMemcpy(pv_output.data(), d_pv_output, pv_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	(int)cudaMemcpy(test100_result.data(), df_output, test100_result.size() * sizeof(float), cudaMemcpyDeviceToHost);

	// 수평방향 미러 패딩
	cout << "horizontal_reflect_padding img" << endl;
	valueCheck2d(p_output, test_H, (test_W + (2 * test_k_radius) - odd));

	// 수평방향 1d ConV
	cout << "Conv1d_horizontal img" << endl;
	valueCheck2d(test100, test_H, test_W);

	// 수직방향 미러 패딩
	cout << "vertical_reflect_padding img" << endl;
	valueCheck2d(pv_output, (test_H + (2 * test_k_radius) - odd), test_W );


	// 수직방향 1d ConV
	cout << "Conv1d_vertical img(final result)" << endl;
	valueCheck2d_6(test100_result, test_H, test_W );

	return 0;
}