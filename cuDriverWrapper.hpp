// CUDA driver API wrapper for vector addition
#ifndef __DRIVERAPI_H__
#define __DRIVERAPI_H__

//#include "vecAdd.h"
#include <iostream>
#include <stdio.h>
#include <vector>
//#include </usr/local/cuda/include/cuda_runtime.h>

#include </usr/local/cuda/include/cuda.h>
#include </home/ayman/CSE/cuda_test/drvapi_error_string.h>
#include </home/ayman/CSE/cuda_test/builtin_types.h>

//extern "C" __global__ void vecAdd(float *a, float *b, float *c, int size);

using namespace std;
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
	if( err != CUDA_SUCCESS)
	{
		printf("Error := %s\n", getCudaDrvErrorString( err) );
		fprintf(stderr, "CUDA Driver API error = %04d from file <%s>, line %i. Error: %s \n",
						err, file, line, getCudaDrvErrorString( err) );
		exit(-1);
	}
}

/*char *module_file = (char*) "vecAdd.ptx";
char *kernel_name = (char*) "vecAdd";*/

class DriverAPI
{
private:
	CUdevice	device;
	CUcontext 	context;
	CUmodule	module;
	CUfunction 	kernel;
	CUstream 	stream;

	CUdeviceptr a_d, b_d, c_d;

	int deviceCount, size;
	float *a_h, *b_h, *c_h;	// host pointers
	char *kernel_name;
	char *module_file;

public:
	std::vector<void *> kernelArgumentVec;

	//constructor
	DriverAPI(int _size)
	{
		size = _size;
		module_file = (char*) "vecAdd.ptx";
		kernel_name = (char*) "vecAdd";

		// initialize CUDA device
		if (cuInit(0) != CUDA_SUCCESS)
		{
			exit(-1);
		}
	}

	// destructor
	~DriverAPI()
	{
		//finalizeCUDA();
	}

	void getDevice()
	{
		// get number of available devices
		//int deviceCount;
		checkCudaErrors( cuDeviceGetCount(&deviceCount) );
		if (deviceCount == 0)
		{
			printf("No devices supporting CUDA. Exiting! \n");
			exit(-1);
		}

		// get handle for only one GPU
		checkCudaErrors( cuDeviceGet(&device, 0) );
	}

	void deviceInfo()
	{
		int major, minor, driverVer, devAttr;
		char deviceName[16];

		checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
		cuDeviceGetName(deviceName, 16, device);
		cuDriverGetVersion(&driverVer);
		cuDeviceGetAttribute(&devAttr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);

		printf("Device Information:\n Name: %s \n CC: %d <-> %d \n Driver: v.%d \n", deviceName, major, minor, driverVer/1000);
		std::cout<<" Compute mode: " << devAttr << std::endl;
	}

	void initCUDA()
	{
/*		char *module_file = (char*) "vecAdd.ptx";
		char *kernel_name = (char*) "vecAdd";*/

		CUresult err = cuInit(0);

		err = cuCtxCreate(&context, 0, device);
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error initializing the CUDA context.\n");
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDetach(context) );
			exit(-1);
		}

		err = cuModuleLoad(&module, module_file);
//		err = cuModuleLoad(&module, "vecAdd.ptx");
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error loading the module %s\n", "vecAdd.ptx");
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDetach(context) );
			exit(-1);
		}

		err = cuModuleGetFunction(&kernel, module, kernel_name);	
		//err = cuModuleGetFunction(&vecAdd, module, "vecAdd");
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDetach(context) );
			exit(-1);
		}

		checkCudaErrors( cuStreamCreate(&stream, 0) );
	}

	void initHostData()
	{
		a_h = new float[size];
		b_h = new float[size];
		c_h = new float[size];

		// intialize array based on type
		for (int i = 0; i < size; i++)
		{
			a_h[i] = i;
			b_h[i] = (i % 5) + 1;
			c_h[i] = 0;
		}
	}

	void setDeviceMemory()
	{
		checkCudaErrors( cuMemAlloc(&a_d, sizeof(float) * size) );
		checkCudaErrors( cuMemAlloc(&b_d, sizeof(float) * size) );
		checkCudaErrors( cuMemAlloc(&c_d, sizeof(float) * size) );
	}

	void setData()
	{
		// copy arrays to device
		checkCudaErrors( cuMemcpyHtoD(a_d, a_h, sizeof(float) * size) );
		checkCudaErrors( cuMemcpyHtoD(b_d, b_h, sizeof(float) * size) );
		checkCudaErrors( cuMemcpyHtoD(c_d, c_h, sizeof(float) * size) );
	}

	void getData()
	{
		// copy array c_d to host array c_h
		checkCudaErrors( cuMemcpyDtoH(c_h, c_d, sizeof(float) * size) );
	}

	void resultPrint(int temp)
	{
		for(int i = 0; i < size; i++)
		{
			printf("%i: \t %f + %f + %i = %f \n", i, a_h[i], b_h[i], temp, c_h[i]);
		}
	}

	void finalizeCUDA()
	{
		// de allocate host memory
		delete[] a_h;
		delete[] b_h;
		delete[] c_h;

		printf("\t Host arrays de-allocated\n");
		checkCudaErrors( cuCtxDetach(context) );
		printf("\t Context dettached\n");
/*		checkCudaErrors( cuStreamDestroy(stream) );
		printf("\t Stream destroyed\n");
*/
	}

	void releaseDeviceMemory()
	{
		checkCudaErrors( cuMemFree(a_d) );
		checkCudaErrors( cuMemFree(b_d) );
		checkCudaErrors( cuMemFree(c_d) );
	}

	inline void setKernelArg(int index, CUdeviceptr *arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = arg;
	}

	inline void setKernelArg(int index, char *arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = arg;
	}

	inline void setKernelArg(int index, int *arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = arg;
	}

	inline void setKernelArg(int index, float *arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = arg;
	}

	void setAllArguments(int *temp)
	{
		//int temp = -100;

		// setup the arguments for the cuda kernel
		kernelArgumentVec.reserve(5);

		setKernelArg(0, temp);
		setKernelArg(1, &a_d);
		setKernelArg(2, &b_d);
		setKernelArg(3, &c_d);
		setKernelArg(4, &size);

//		printf("--args: [0] [1] [2] [3] [4] %i  %i  %i  %i  %i \n", kernelArgumentVec[0], kernelArgumentVec[1], kernelArgumentVec[2], kernelArgumentVec[3], kernelArgumentVec[4]);
	}

	void runKernel2(std::vector<void *>& arguments)
	{
//		printf("std::vec A: [0] [1] [2] [3] [4] %i  %i  %i  %i  %i \n", arguments[0], arguments[1], arguments[2], arguments[3], arguments[4]);

		dim3 block 	= dim3(32, 1, 1);
		dim3 grid 	= dim3((size + block.x - 1) / block.x, 1, 1);

		// launch the kernel
		checkCudaErrors( cuLaunchKernel(kernel,
										grid.x,
										grid.y,
										grid.z,  // Nx1x1 blocks
										block.x,
										block.y,
										block.z, // 1x1x1 threads
										0,
										stream,
										&arguments[0],
										0) );
		//checkCudaErrors( cuLaunchKernel(vecAdd, 2, 1, 1,  // Nx1x1 blocks
		//						32, 1, 1,            // 1x1x1 threads
		//						0, stream, args, 0) );
	}
};

#endif	// __DRIVERAPI_H__