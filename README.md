# Gaussian Filter

## Environment
- Windows 10
- CUDA 11.1
- TensorRT 8.0.1.6
- Cudnn 8.2.1
***
## Cudnn Convolution APIs
Image height :: 1000   
Image width :: 2000   
sigma :: 3   
ksize in GPU :: 10   
ksize in CPU :: 11   

Given ksize value is even number   
0.0367883
0.0606536
0.0894848
0.118137
0.139563
0.147535
0.139563
0.118137
0.0894848
0.0606536

1. CUDA Kernel Gaussian filtering   
avg_dur_time =  0.487 [msec]

2. OpenCV Gaussian filtering   
avg_dur_time = 1.524 [msec] 

3. Cpp function Gaussian filtering   
avg_dur_time = 145.526 [msec] 

4. CuDNN function Gaussian filtering   
avg_dur_time = 41.380 [msec]

***