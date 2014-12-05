//main.cpp

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include "common.h"
#include "classes.hpp"
#include "CudaCompile.hpp"
#include "cuDriverWrapper.hpp"

using namespace std;

//#include </usr/local/cuda/include/cuda_runtime.h>
//#include </usr/local/cuda/include/cuda.h>
//#include "vecAdd.h"

extern "C" void gpu_memAlloc(int **a, int **b, int **c, int size);
extern "C" void gpu_setData(int *dst, int *src, int size);
extern "C" void gpu_getData(int *src, int *dst, int size);
extern "C" void gpu_addVectors(int *a_d, int *b_d, int *c_d, int size);
extern "C" void gpu_memRelease(int *a_d, int *b_d, int *c_d);

class GpuVectorAddition
{
//private:
public:
	int *a_h, *b_h, *c_h, *a_d, *b_d, *c_d, size;	// number of array elements

	// member functions
	GpuVectorAddition(int _size);

	void init();
	void addData();
	void result();

	~GpuVectorAddition();
};

GpuVectorAddition::GpuVectorAddition(int _size)
{
	size = _size;
}

GpuVectorAddition::~GpuVectorAddition()
{
	delete[] a_h;
	delete[] b_h;
	delete[] c_h;
}
void GpuVectorAddition::init()
{
	a_h = new int[size];
	b_h = new int[size];
	c_h = new int[size];

	gpu_memAlloc(&a_d, &b_d, &c_d, size);

	for (int i = 0; i < size; i++)
	{
		a_h[i] = i;
		b_h[i] = (i % 5) + 1;
		c_h[i] = 0;
	}

	gpu_setData(a_d, a_h, size);
	gpu_setData(b_d, b_h, size);
	gpu_setData(c_d, c_h, size);
}

void GpuVectorAddition::addData()
{
	gpu_addVectors(a_d, b_d, c_d, size);
}

void GpuVectorAddition::result()
{
	gpu_getData(c_h, c_d, size);
	gpu_getData(b_h, b_d, size);

	for(int i = 0; i < size; i++)
	{
		printf("%i: \t %i + %i = %i \n", i, a_h[i], b_h[i], c_h[i]);
	}

	gpu_memRelease(a_d, b_d, c_d);
}

int main(int argc, char **argv)
{
	size_t elems = atoi(argv[1]);
	int grid_x = atoi(argv[2]);
	int temp = atoi(argv[3]);
	unsigned int dim = atoi(argv[4]);
	char *kernelName = argv[5];

/*	int n = 512;//atoi(argv[1]);
	int temp = atoi(argv[1]);
	unsigned int dim = atoi(argv[2]);	///< dimnesion must be greater than zero	*/

	// CPU computation
	CProgram cCompileCuda(kernelName);
	CMain *cpu_class = new CMain(elems);

	cpu_class -> memAlloc();	// alloc input arrays on host (CPU)
	cpu_class -> initialize();	// initialize arrays
	cpu_class -> addVectors();	// CPU computation
	//cpu_class -> argSet();
	//cpu_class -> argPrint( (*cpu_class).arguments );
	cout << "CPU:"<<endl;		// print results
	cpu_class -> resultPrint();
	cpu_class -> memRelease();	// free CPU arrays

/*	// GPU computations, CUDA Runtime API class
	GpuVectorAddition P = GpuVectorAddition(n);
	P.init();
	P.addData();
	cout << endl << "GPU:"<< endl;
	P.result();*/

	// Compile cuda source kernel
	//cCompileCuda.CProgram(kernelName);


	// GPU CUDA Driver API wrapper class examṕle
	DriverAPI *gpu_DAPI_class = new DriverAPI(elems);
	size_t *globalSize = new size_t(dim);
	size_t *localSize = new size_t(3);

	localSize[0] = 32;
	localSize[1] = 32;
	localSize[2] = 1;

	gpu_DAPI_class -> getDevice();
	gpu_DAPI_class -> deviceInfo();

	// compile cuda kernel to ptx file
	cCompileCuda.createCompileCommand((*gpu_DAPI_class).major);

	gpu_DAPI_class -> initCUDA();
	gpu_DAPI_class -> initContextModule();
	gpu_DAPI_class -> initHostData();
	gpu_DAPI_class -> setDeviceMemory();
	gpu_DAPI_class -> setData();
	gpu_DAPI_class -> setAllArguments(temp);
	gpu_DAPI_class -> runKernel2(dim, localSize, globalSize, grid_x, (*gpu_DAPI_class).kernelArgumentVec );
	gpu_DAPI_class -> getData();
	gpu_DAPI_class -> resultPrint(temp);
	gpu_DAPI_class -> releaseDeviceMemory();
	gpu_DAPI_class -> finalizeCUDA();
	printf("free cuda context and memory..\n Alles klar! \n");
	
	delete gpu_DAPI_class;
	delete cpu_class;
	delete globalSize;
	delete localSize;

	//int ret_val = system("gedit &");

	//printf("return val: %i\n", ret_val);

	return 1;
}



// nvcc compile works in this way
/*
nvcc -x cu -arch=sm_20 -I. -dc main.cpp -o main.o 
nvcc -x cu -arch=sm_20 -I. -dc classes.cpp -o classes.o 
nvcc -x cu -arch=sm_20 -I. -dc cuDriverWrapper.hpp -o cuDriverWrapper.o
nvcc -arch=sm_20 main.o classes.o cuDriverWrapper.o -o app

*/