#include </usr/local/cuda/include/cuda.h>
//#include "vecAdd.h"

extern "C" __global__ void vecAdd(int *a, int *b, int *c, int size)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx < size)
	{
		c[idx] = a[idx] + b[idx];
		//printf("%i\n",idx );
	}
}