
#include </usr/local/cuda/include/cuda.h>
#include "lbm_header.h"

extern "C" __global__ void vecAdd(const int temp, float *a, float *b, float *c, const size_t size)
{
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_z = threadIdx.z + blockDim.z * blockIdx.z;

	int idx_xy = idx_y * (blockDim.x * gridDim.x) + idx_x;
	int idx = idx_z * (blockDim.x * gridDim.x + blockDim.y * gridDim.y) + idx_xy;

	if(idx < size)
	{
		c[idx] = a[idx] + b[idx] + (T)temp;
		// c[idx] = (T)temp;
	}
}