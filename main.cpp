//main.cpp

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "classes.hpp"
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
	int n = atoi(argv[1]);
	//printf("begin..\n");
	CMain *cpu_class = new CMain(n);

	cpu_class -> memAlloc();	// alloc input arrays on host (CPU)
	cpu_class -> initialize();	// initialize arrays
	cpu_class -> addVectors();	// CPU computation

	// print result
	cout << "CPU:"<<endl;
	cpu_class -> resultPrint();
	cout << endl;

	// free CPU arrays
	cpu_class -> memRelease();

	// GPU computations
	GpuVectorAddition P = GpuVectorAddition(n);
	P.init();
	P.addData();
	cout << "GPU:"<<endl;
	P.result();

#if 1
	DriverAPI *gpu_DAPI_class = new DriverAPI(n);

	gpu_DAPI_class -> getDevice();
	gpu_DAPI_class -> deviceInfo();

	gpu_DAPI_class -> initCUDA();
	printf("init done..\n");

	gpu_DAPI_class -> initHostData();
	printf("host data initialized..\n");

	gpu_DAPI_class -> setDeviceMemory();
	printf("GPU memory set..\n");

	gpu_DAPI_class -> setData();
	printf("GPU data set..\n");

	gpu_DAPI_class -> runKernel();
	printf("kernel executed..\n");

	gpu_DAPI_class -> getData();
	printf("copy back results..\n");

	gpu_DAPI_class -> resultPrint();

	gpu_DAPI_class -> releaseDeviceMemory();
	printf("release device memory..\n");

	gpu_DAPI_class -> finalizeCUDA();
	printf("free cuda context and memory..\n Alles klar!");
	
	delete gpu_DAPI_class;
#endif

	delete cpu_class;

	return 1;
}



// nvcc compile works in this way
/*
nvcc -x cu -arch=sm_20 -I. -dc main.cpp -o main.o 
nvcc -x cu -arch=sm_20 -I. -dc classes.cpp -o classes.o 
nvcc -x cu -arch=sm_20 -I. -dc cuDriverWrapper.hpp -o cuDriverWrapper.o
nvcc -arch=sm_20 main.o classes.o cuDriverWrapper.o -o app

*/