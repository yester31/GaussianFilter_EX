#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "Gaussian.h"


// Filter container
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}

// 수평 방향 REFLECTIVE 패딩 (미러 패딩)
__global__ void horizontal_reflect_padding_kernel(float* __restrict__ output, float* __restrict__ input, int H, int W, int k_radius, int tcount, int odd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tcount) return;

    int wp = ((W + k_radius * 2) - odd);
    int pw_idx = tid % wp;
    int h_idx = tid / wp;
    int o_idx = h_idx * wp + pw_idx;

    // front padding
    if (o_idx % wp < k_radius - odd) {
        int i_idx = k_radius - odd - o_idx % wp + h_idx * W;
        output[o_idx] = input[i_idx];
    }// back padding
    else if (o_idx % wp >= W + k_radius - odd) {
        int i_idx2 = 2 * W - (o_idx % wp) + k_radius - odd + h_idx * W - 2;
        output[o_idx] = input[i_idx2];
    }
    else {
        if (h_idx == 0) {
            int ii_idx = o_idx - (k_radius - odd);
            output[o_idx] = input[ii_idx];
        }
        else {
            int ii_idx = o_idx - (k_radius - odd) - h_idx * (k_radius * 2 - odd);
            output[o_idx] = input[ii_idx];
        }
    }
}
extern "C" void horizontal_reflect_padding(float* d_output, float* d_data, int H, int W, int k_radius, int odd) {

    int tcount = H * (W + k_radius * 2 - odd);
    int thread_X = 512;
    int blockX = (tcount + thread_X - 1) / thread_X;

    dim3 grid(blockX, 1, 1);
    dim3 block(thread_X, 1, 1);

    horizontal_reflect_padding_kernel << <grid, block >> > (d_output, d_data, H, W, k_radius, tcount, odd);
}


// 수직방향 REFLECTIVE 패딩 (미러 패딩)
__global__ void vertical_reflect_padding_kernel(float* __restrict__ output, float* __restrict__ input, int H, int W, int k_radius, int tcount, int odd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tcount) return;

    int w_idx = tid % W;
    int h_idx = tid / W;
    int o_idx = h_idx * W + w_idx;

    if (h_idx < k_radius - odd) {
        int i_idx = (k_radius - odd - h_idx) * W + w_idx;
        output[o_idx] = input[i_idx];
    }
    else if (h_idx > H + k_radius - (1 + odd)) {
        int i_idx2 = (H * 2 + (k_radius - odd) - (h_idx + 2))* W + w_idx;
        output[o_idx] = input[i_idx2];
    }
    else {
        int ii_idx = o_idx - ((k_radius - odd) * W);
        output[o_idx] = input[ii_idx];
    }

}
extern "C" void vertical_reflect_padding(float* d_output, float* d_data, int H, int W, int k_radius, int odd) {

    int tcount = W * (H + (k_radius * 2) - odd);
    int thread_X = 512;
    int blockX = (tcount + thread_X - 1) / thread_X;

    dim3 grid(blockX, 1, 1);
    dim3 block(thread_X, 1, 1);

    vertical_reflect_padding_kernel << <grid, block >> > (d_output, d_data, H, W, k_radius, tcount, odd);
}

// 1D 수평 방향 컨볼류션
__global__ void conv1d_horizontal_kernel(float* __restrict__ output, float* __restrict__ input, int H, int W, int k_radius, int tcount, int odd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tcount) return;

    int w_idx = tid % W;
    int h_idx = tid / W;
    int o_idx = h_idx * W + w_idx;
    int i_idx = h_idx * (W + k_radius * 2 - odd) + w_idx;

    if (h_idx >= H) return;

    float sum = 0;

    for (int j = 0; j <= k_radius * 2; j++) {
        sum += c_Kernel[j] * input[i_idx + j];
    }

    output[o_idx] = sum;
}

extern "C" void conv1d_horizontal(float* d_output, float* d_data, int H, int W, int k_radius, int odd) {

    int tcount = H * (W + k_radius * 2);
    int thread_X = 512;
    int blockX = (tcount + thread_X - 1) / thread_X;

    dim3 grid(blockX, 1, 1);
    dim3 block(thread_X, 1, 1);

    conv1d_horizontal_kernel << <grid, block >> > (d_output, d_data, H, W, k_radius, tcount, odd);
}

__global__ void conv1d_vertical_kernel(float* __restrict__ output, float* __restrict__ input, int H, int W, int k_radius, int tcount, int odd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tcount) return;

    int w_idx = tid % W;
    int h_idx = tid / W;
    int o_idx = h_idx * W + w_idx;

    if (h_idx >= H) return;

    float sum = 0;

    for (int j = 0; j <= k_radius * 2 - odd; j++) {
        sum += c_Kernel[j] * input[o_idx + j * W];
    }

    output[o_idx] = sum;

}
extern "C" void conv1d_vertical(float* d_output, float* d_data, int H, int W, int k_radius, int odd) {

    int tcount = W * (H + k_radius * 2 - odd);

    int thread_X = 512;
    int blockX = (tcount + thread_X - 1) / thread_X;

    dim3 grid(blockX, 1, 1);
    dim3 block(thread_X, 1, 1);

    conv1d_vertical_kernel << <grid, block >> > (d_output, d_data, H, W, k_radius, tcount, odd);
}