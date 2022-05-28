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

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td></td>
			<td><strong>CUDA Kernel</strong></td><td><strong>OpenCV</strong></td><td><strong>CuDNN</strong></td><td><strong>Cpp function</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>0.487 ms</td>
			<td>1.524 ms </td>
			<td>41.380 ms</td>
			<td>145.526 ms</td>
		</tr>
	</tbody>
</table>

***