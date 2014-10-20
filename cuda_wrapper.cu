#include <cuda_runtime.h>
#include <iostream>
#include <string>

using namespace std;

#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line)
{
	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		std::cout << std::endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << std::endl;
		exit(1);
	}
}

// CUDA add float
__device__ float Add(int a, int b)
{
	return a+b;
}

// CUDA array adding
__global__ void AddArrays(int *a, int *b, int *c, int size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < size)
	{
		c[idx] = Add(a[idx], b[idx]);
	}
}

extern "C" void gpu_memAlloc(int **a, int **b, int **c, int size)
{
	cudaMalloc(a, sizeof(int) * size);	CUDA_CHECK;
	cudaMalloc(b, sizeof(int) * size);	CUDA_CHECK;
	cudaMalloc(c, sizeof(int) * size);	CUDA_CHECK;
}
//copying data from host to device
extern "C" void gpu_setData(int *dst_d, int *src_h, int size)
{
	cudaMemcpy(dst_d, src_h, sizeof(int) * size, cudaMemcpyHostToDevice);	CUDA_CHECK;
}

extern "C" void gpu_getData(int *dst_h, int *src_d, int size)
{
	cudaMemcpy(dst_h, src_d, sizeof(int) * size, cudaMemcpyDeviceToHost);	CUDA_CHECK;
}

//Addition function
//number of thread and block is set before call Kernel
extern "C" void gpu_addVectors(int *a_d, int *b_d, int *c_d, int size)
{
	dim3 block = dim3(32, 1, 1);
	dim3 grid = dim3((size + block.x - 1) / block.x, 1, 1);
	{
		AddArrays<<<grid, block>>>(a_d, b_d, c_d, size);	
	}
	CUDA_CHECK;
}
//read data back to host
extern "C" void gpu_memRelease(int *a_d, int *b_d, int *c_d)
{
	cudaFree(a_d);	CUDA_CHECK;
	cudaFree(b_d);	CUDA_CHECK;	
	cudaFree(c_d);	CUDA_CHECK;
}

