#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cudnn.h>

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
	Mat original_src = imread("test.jpg", IMREAD_GRAYSCALE);
	Mat src;
	resize(original_src, src, Size(2000,1000));
	// 원본 사진
	//imshow("Original Image", src);

	Mat src_gpu = src;
	Mat src_cpu = src;
	Mat opencv_src = src;
	Mat src_cudnn = src;

	int H = src.rows;
	int W = src.cols;
	cout << "Image height :: " << H << endl;
	cout << "Image width :: "  << W << endl << endl;

	float sigma = 3.0;
	int k_radius = KERNEL_LENGTH / 2;
	int ksize = KERNEL_LENGTH;

	cout << "sigma :: " << sigma << endl;
	cout << "ksize in GPU :: " << ksize << endl;

	int odd = 0;
	if (KERNEL_LENGTH % 2 == 0) {// even number, 짝수면
		ksize += 1;
		odd = 1;
		cout << "ksize in CPU :: " << ksize << endl;
		cout << "Given ksize value is even number " << endl << endl;
	}
	else {
		cout << "ksize in CPU :: " << ksize << endl;
		cout << "Given ksize value is odd number " << endl << endl;
	}

	//cudaSetDevice(0);

	// 필터를 담을 벡터 생성
	vector<float> filter(ksize);

	// 가우시안 필터 생성
	gaussianFilter1d(filter, sigma, k_radius, odd);

	// vector로 변경한 필터 값 체크 
	//filter_check(filter);

	float *h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));

	for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
	{
		h_Kernel[i] = filter[i];
		cout << filter[i] << endl;
	}

	setConvolutionKernel(h_Kernel);

	vector<float> input(W * H);
	vector<float> temp(W * H);
	vector<float> output(W * H);

	// mat 형식 이미지를 벡터로 변경
	mat_to_vector2(input, src);

	//cout << "Original img pixel value" << endl;
	//valueCheck2d(input, H, W);

	float* d_data;	   // 이미지
	(int)cudaMalloc(&d_data, input.size() * sizeof(float));
	(int)cudaMemcpy(d_data, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);


	vector<float> p_output(H * (W + 2*k_radius - odd));
	float* d_p_output;
	(int)cudaMalloc(&d_p_output, p_output.size() * sizeof(float));

	vector<float> pv_output((H + 2*k_radius - odd) * W);
	float* d_pv_output;
	(int)cudaMalloc(&d_pv_output, pv_output.size() * sizeof(float));

	float* d_output;	// 수평 conv1d 결과값
	(int)cudaMalloc(&d_output, input.size() * sizeof(float));

	float* df_output;	// 수직 conv1d 결과값 , 최종 결과 값
	(int)cudaMalloc(&df_output, output.size() * sizeof(float));


	cout << endl << "Running CUDA KERNEL using GPU ..." << endl << endl;

	uint64_t total_time = 0;
	for (int iIdx = 0; iIdx < ITER; iIdx++) {
		uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

		horizontal_reflect_padding(d_p_output, d_data, H, W, k_radius, odd);
		conv1d_horizontal(d_output, d_p_output, H, W, k_radius, odd);
		vertical_reflect_padding(d_pv_output, d_output, H, W, k_radius, odd);
		conv1d_vertical(df_output, d_pv_output, H, W, k_radius, odd);

		total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;
	}

	//(int)cudaMemcpy(p_output.data(), d_p_output, p_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//(int)cudaMemcpy(temp.data(), d_output, temp.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//(int)cudaMemcpy(pv_output.data(), d_pv_output, pv_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	(int)cudaMemcpy(output.data(), df_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

	float checksum = 0;
	for (auto d : output) checksum += fabs((float)d);

	printf("======================================================\n\n");
	printf("1. CUDA Kernel Gaussian filtering dur_time = %6.3f[msec]  checksum = %.6f \n\n", total_time / 1000.f / ITER , checksum);
	printf("======================================================\n\n");

	array_to_mat(src_gpu, output);
	//imshow("CUDA KERNEL", src_gpu);

	// 수평방향 미러 패딩
	//cout << "horizontal_reflect_padding img" << endl;
	//valueCheck2d(p_output, H, (W + 2 * k_radius - odd) );

	// 수평방향 1d ConV
	//cout << "Conv1d_horizontal img" << endl;
	//valueCheck2d(temp, H , W);

	// 수직방향 미러 패딩
	//cout << "vertical_reflect_padding img" << endl;
	//valueCheck2d(pv_output, (H + 2 * k_radius - odd), W);

	// 수직방향 1d ConV
	//cout << "Conv1d_vertical img(final result)" << endl;
	//valueCheck2d(output, H, W);

	//================================================================================================
	// openCV 함수 이용(cpu)
	cout << "Running OpenCV function using CPU ..." << endl << endl;

	Mat dst;

	uint64_t total_time3 = 0;
	for (int iIdx = 0; iIdx < ITER; iIdx++) {
		uint64_t start_time3 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

		GaussianBlur(opencv_src, dst, Size(ksize, ksize), (float)sigma);

		total_time3 += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time3;
	}

	// mat 형식 이미지를 벡터로 변경
	vector<float> output_opencv(W * H);
	mat_to_vector2(output_opencv, dst);

	//valueCheck2d(output_opencv, H, W);

	float checksum3 = 0;
	for (auto d : output_opencv) checksum3 += fabs((float)d);

	printf("======================================================\n\n");
	printf("2. OpenCV Gaussian filtering avg_dur_time = %6.3f[msec]  checksum3 = %.6f \n\n", total_time3 / 1000.f / ITER, checksum3);
	printf("======================================================\n");

	//imshow("OpenCV", dst);

	//================================================================================================


	//================================================================================================
	// Cpu 함수 이용

	
	// gaussian 2d kernel 생성 함수
	vector<float> filter2d(KERNEL_LENGTH * KERNEL_LENGTH);
	// 2d 가우시안 필터 생성
	gaussianFilter2d(filter2d, sigma, k_radius, odd);

	
	// 필터값 확인
	//valueCheck2d_6(filter2d, KERNEL_LENGTH, KERNEL_LENGTH);
	
	// cpu gaussian 결과 값 담을 벡터
	vector<float> output_cpu(W * H);
	// reflect padding 결과 값 을 담을 벡터	
	vector<float> data_pv(((H + k_radius * 2) - odd) * ((W + k_radius * 2) - odd));

	cout << "Running  cpp function using CPU ..." << endl << endl;

	uint64_t total_time2 = 0;
	for (int iIdx = 0; iIdx < ITER; iIdx++) {
		uint64_t start_time2 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		
		output_cpu.assign(W*H,0);
		// reflect padding
		reflectivePadding(data_pv, input, H, W, k_radius, odd);
		// filtering, 컨볼류션
		convolution(output_cpu, data_pv, filter2d, KERNEL_LENGTH, 1, 1, 1, H + k_radius * 2 - odd, W + k_radius * 2 - odd, 1);

		total_time2 += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time2;
	}

	float checksum2 = 0;
	for (auto d : output_cpu) checksum2 += fabs((float)d);

	printf("======================================================\n\n");
	printf("3. cpp function Gaussian filtering dur_time = %6.3f[msec]  checksum2 = %.6f  \n\n", total_time2 / 1000.f / ITER, checksum2);
	printf("======================================================\n\n");

	// 계산 값 확인
	//valueCheck2d(output_cpu, H, W);
	//array_to_mat(src_cpu, output_cpu);
	//imshow("CPP function", src_cpu);
	

	//================================================================================================
	// CUDNN 함수 이용

	cout << "Running  CUDNN function using GPU ..." << endl << endl;


	int status = 0;
	//cudaStream_t stream;
	cudnnHandle_t cudnnHandle;
	//status |= (int)cudaStreamCreate(&stream);
	status |= (int)cudnnCreate(&cudnnHandle);
	//status |= (int)cudnnSetStream(cudnnHandle, stream);

	vector<float> weight(KERNEL_LENGTH * KERNEL_LENGTH);
	float* d_weight; // device weight 
	status |= (int)cudaMalloc(&d_weight, filter2d.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_weight, filter2d.data(), filter2d.size() * sizeof(float), cudaMemcpyHostToDevice);

	vector<float> bias(1);
	inititalizedDataOne(bias);

	float* d_bias; // device bias
	status |= (int)cudaMalloc(&d_bias, bias.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);

	vector<float> output_cudnn(H * W);
	float* d_output_cudnn; // device output data
	status |= (int)cudaMalloc(&d_output_cudnn, output_cudnn.size() * sizeof(float));
	status |= (int)cudaMemset(d_output_cudnn, 0, output_cudnn.size() * sizeof(float));

	int Hp = H + 2 * k_radius - odd;
	int Wp = W + 2 * k_radius - odd;

	vector<float> pv_temp(Hp * Wp);
	float* d_pv_temp;
	status |= (int)cudaMalloc(&d_pv_temp, pv_temp.size() * sizeof(float));


	cudnnTensorDescriptor_t xdesc;
	status |= (int)cudnnCreateTensorDescriptor(&xdesc);
	status |= (int)cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, Hp, Wp);

	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status |= (int)cudnnCreateFilterDescriptor(&wdesc);
	status |= (int)cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, KERNEL_LENGTH, KERNEL_LENGTH);

	cudnnConvolutionDescriptor_t conv_desc;
	status |= (int)cudnnCreateConvolutionDescriptor(&conv_desc);
	status |= (int)cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION

	cudnnTensorDescriptor_t ydesc;
	status |= (int)cudnnCreateTensorDescriptor(&ydesc);
	status |= (int)cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, H, W);

	cudnnConvolutionFwdAlgo_t cudnn_algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
	float one = 1.f, zero = 0.f;
	status |= (int)cudnnGetConvolutionForwardAlgorithm(cudnnHandle, xdesc, wdesc, conv_desc, ydesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &cudnn_algo);

	size_t workspace_size;
	status |= (int)cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xdesc, wdesc, conv_desc, ydesc, cudnn_algo, &workspace_size);
	float* d_workspace = nullptr;
	status |= (int)cudaMalloc(&d_workspace, workspace_size);

	uint64_t total_time4 = 0;
	for (int idx = 0; idx < ITER; idx++) {
		uint64_t start_time4 = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

		horizontal_reflect_padding(d_p_output, d_data, H, W, k_radius, odd);
		vertical_reflect_padding(d_pv_temp, d_p_output, H, Wp, k_radius, odd);

		cudnnConvolutionForward(
				cudnnHandle,
				&one, 
				xdesc, d_pv_temp,
				wdesc, d_weight,
				conv_desc, cudnn_algo,
				d_workspace, workspace_size,
				&zero,
				ydesc, 
				d_output_cudnn);

		//status |= (int)cudaStreamSynchronize(stream);
		total_time4 += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time4;
	}
	

	status |= (int)cudaMemcpy(output_cudnn.data(), d_output_cudnn, output_cudnn.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//valueCheck2d(output_cudnn, H, W);

	//status |= (int)cudaMemcpy(pv_temp.data(), d_pv_temp, pv_temp.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//valueCheck2d(pv_temp, Hp, Wp);

	//status |= (int)cudaMemcpy(weight.data(), d_weight, weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//valueCheck2d(weight, KERNEL_LENGTH, KERNEL_LENGTH);	// 출력값 확인

	float checksum4 = 0;
	for (auto d : output_cudnn) checksum4 += fabs((float)d);
	printf("======================================================\n\n");

	printf("4. cuDNN function Gaussian filtering avg_dur_time=%6.3f[msec] checksum4=%.6f \n\n", total_time4 / 1000.f / ITER, checksum4);
	printf("======================================================\n\n");

	//array_to_mat(src_cudnn, output_cudnn);
	//imshow("CUDNN Conv", src_cudnn);


	status |= (int)cudaFree(d_output_cudnn);
	status |= (int)cudaFree(d_weight);

	//================================================================================================

	waitKey();

	cudaFree(h_Kernel);
	cudaFree(d_data);
	cudaFree(d_p_output);
	cudaFree(d_pv_output);
	cudaFree(d_output);
	cudaFree(df_output);

	cudnnDestroy(cudnnHandle);

	return 0;
}