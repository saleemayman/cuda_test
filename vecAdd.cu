
#include </usr/local/cuda/include/cuda.h>
#include "lbm_header.h"

extern "C" __global__ void vecAdd(int temp, float *a, float *b, float *c, size_t size)
{
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_z = threadIdx.z + blockDim.z * blockIdx.z;

	int idx_xy = idx_y * (blockDim.x * gridDim.x) + idx_x;
	int idx = idx_z * (blockDim.x * gridDim.x + blockDim.y * gridDim.y) + idx_xy;

	// try shared memory allocation as per lbm_beta.cu
	__shared__ T dd_buf[1][LOCAL_WORK_GROUP_SIZE];
	extern __shared__ T *dd_buf_lid;


	if(idx < size)
	{
		dd_buf[1][idx] = a[idx];
		dd_buf_lid = &dd_buf[1][idx];

		//c[idx] = a[idx] + b[idx];
		c[idx] = *dd_buf_lid;
		//c[idx] = (T)(DOMAIN_CELLS + DOMAIN_CELLS_X + DOMAIN_CELLS_Y);
	}
}