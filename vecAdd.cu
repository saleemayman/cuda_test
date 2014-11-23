#include </usr/local/cuda/include/cuda.h>
//#include "vecAdd.h"

extern "C" __global__ void vecAdd(int temp, float *a, float *b, float *c, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{
		c[idx] = (float)temp + a[idx] + b[idx];
	}
}