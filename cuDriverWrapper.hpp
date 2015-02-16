// CUDA driver API wrapper for vector addition
#ifndef __DRIVERAPI_H__
#define __DRIVERAPI_H__

//#include "vecAdd.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
//#include </usr/local/cuda/include/cuda_runtime.h>

#include "common.h"

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

class DriverAPI
{
private:
	CUdevice	device;
	CUcontext 	context;
	CUmodule	module;
	CUfunction 	kernel;
	CUstream 	stream;

	CUdeviceptr a_d, b_d, c_d;
	CUresult err;

	int deviceCount;
	size_t size;
	float *a_h, *b_h, *c_h;	// host pointers
	std::string module_file, kernel_name;

//	char *module_file = (char*) "vecAdd.ptx";
//	char *kernel_name = (char*) "vecAdd";

public:
	int major, minor, driverVer, devAttr;
	std::vector<void *> kernelArgumentVec;

	//constructor
	DriverAPI(size_t _size)
	{
		size = _size;
		module_file = "vecAdd.ptx";
		kernel_name = "vecAdd";

		// initialize CUDA driver API
		initCUDA();
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
		// initialize CUDA device
		if (cuInit(0) != CUDA_SUCCESS)
		{
			exit(-1);
		}
	}

	void initContextModule()
	{
		createContext();
		attachModule(module_file);
		attachFunction( kernel_name.c_str() );
	}

	void createContext()
	{
		err = cuCtxCreate(&context, 0, device);
		cout << "context: " << err << endl;
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error creating the CUDA context.\n");
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDestroy(context) );
			exit(-1);
		}
	}

	void attachModule(const std::string &file)	//const char &module_file
	{
		moduleMember( file.c_str() );
	}

	void moduleMember(const char *file)
	{
		err = cuModuleLoad(&module, file);	//	err = cuModuleLoad(&module, "vecAdd.ptx");
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error loading the module %s\n", file);
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDestroy(context) );
			exit(-1);
		}
	}

	void attachFunction(const char *file)
	{
		err = cuModuleGetFunction(&kernel, module, file);	//err = cuModuleGetFunction(&vecAdd, module, "vecAdd");
		if (err != CUDA_SUCCESS)
		{
			fprintf(stderr, "* Error getting kernel function %s\n", file);
			checkCudaErrors( err);
			checkCudaErrors( cuCtxDestroy(context) );
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
		for (size_t i = 0; i < size; i++)
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
		printf("c[i]: \n");
		for(size_t i = 0; i < size; i++)
		{
/*			if (i == c_h[i])
			{
				printf("%i", 1);
			}
			else
			{
				printf("\t %i \t", (int)c_h[i]);
			}
*/
			//printf("%i ", (int)c_h[i]);
			printf("%lu: \t %f + %f + %i = %f \n", i, a_h[i], b_h[i], temp, c_h[i]);
		}
//		printf("c[%lu]: %f\n", (size - 1), c_h[size - 1]);
//		printf("\n");
	}

	void finalizeCUDA()
	{
		// de allocate host memory
		delete[] a_h;
		delete[] b_h;
		delete[] c_h;

		printf("\t Host arrays de-allocated\n");
		checkCudaErrors( cuCtxDestroy(context) );
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

	inline void setKernelArg(int index, CUdeviceptr &arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = &arg;
	}

	inline void setKernelArg(int index, char &arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = &arg;
	}

	inline void setKernelArg(int index, int &arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = &arg;
	}

	inline void setKernelArg(int index, float &arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = &arg;
	}
	inline void setKernelArg(int index, size_t &arg)
	{
//		kernelArgumentVec.push_back(arg);
		kernelArgumentVec[index] = &arg;
	}

	void setAllArguments(int &temp)
	{
		//int temp = -100;

		// setup the arguments for the cuda kernel
		kernelArgumentVec.reserve(5);

		setKernelArg(0, temp);
		setKernelArg(1, a_d);
		setKernelArg(2, b_d);
		setKernelArg(3, c_d);
		setKernelArg(4, size);

//		printf("--args: [0] [1] [2] [3] [4] %i  %i  %i  %i  %i \n", kernelArgumentVec[0], kernelArgumentVec[1], kernelArgumentVec[2], kernelArgumentVec[3], kernelArgumentVec[4]);
	}

	void setGridAndBlockSize(	dim3 &grid, dim3 &block, unsigned int work_dim,
								const int grid_size_x, const size_t total_elems, 
								size_t *local_work_size, size_t *global_work_size
	)
	{
		size_t threads_per_block;

		// if no work-group size specified set it to default defined in lbm_defaults.h
		if (local_work_size == NULL)
		{
			if (work_dim == 1)
			{
				block = dim3(LOCAL_WORK_GROUP_SIZE, 1 ,1);
				global_work_size[0] = total_elems;
				
				grid = dim3((total_elems + block.x - 1)/block.x, 1, 1);
			}
			else if(work_dim == 2)
			{
				block = dim3(LOCAL_WORK_GROUP_SIZE/2, 2, 1);

				global_work_size[0] = grid_size_x * LOCAL_WORK_GROUP_SIZE;
				global_work_size[1] = total_elems/global_work_size[0];

				grid = dim3(grid_size_x, 
							(total_elems + global_work_size[0] - 1)/global_work_size[0], 1);
			}
			else
			{
				block = dim3(LOCAL_WORK_GROUP_SIZE/4, 2 ,2);
				
				global_work_size[0] = grid_size_x * LOCAL_WORK_GROUP_SIZE;
				global_work_size[1] = total_elems/global_work_size[0];
				global_work_size[2] = 1;

				grid = dim3(grid_size_x, 
							(total_elems + global_work_size[0] - 1)/global_work_size[0], 1);
			}
		}
		else
		{
			block = dim3(local_work_size[0], local_work_size[1], local_work_size[2]);
			threads_per_block = block.x * block.y * block.z;

			if (work_dim == 1)
			{
				global_work_size[0] = total_elems;
				grid = dim3((total_elems + threads_per_block - 1)/threads_per_block, 1, 1);
			}
			else if(work_dim == 2)
			{
				global_work_size[0] = grid_size_x * threads_per_block;
				global_work_size[1] = total_elems/global_work_size[0];

				grid = dim3(grid_size_x, 
							(total_elems + global_work_size[0] - 1)/global_work_size[0], 1);
			}
			else
			{
				global_work_size[0] = grid_size_x * threads_per_block;
				global_work_size[1] = total_elems/global_work_size[0];
				global_work_size[2] = 1;

				grid = dim3(grid_size_x, 
							(total_elems + global_work_size[0] - 1)/global_work_size[0], 1);
			}
		}
	}

	void runKernel2(unsigned int work_dim, size_t *local_work_size, 
					size_t *global_work_size, const int grid_size_x,
					std::vector<void *>& arguments
	)
	{
		dim3 block, grid;// 	= dim3(32, 1, 1);

		setGridAndBlockSize(grid, block, work_dim, grid_size_x, size, local_work_size, global_work_size);

		/*grid = dim3(4, 4, 1);
		block = dim3(32, 32, 1);*/
		printf("grid: [%d %d %d], block: [%d %d %d] \n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
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
	}
};

#endif	// __DRIVERAPI_H__