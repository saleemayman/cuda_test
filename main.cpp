//main.cpp

#include <iostream>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include <mpi.h>

#include "common.h"
// #include "lbm_header.h"
#include "classes.hpp"
#include "CudaCompile.hpp"
#include "cuDriverWrapper.hpp"


using namespace std;

// extern "C" void gpu_memAlloc(int **a, int **b, int **c, int size);
// extern "C" void gpu_setData(int *dst, int *src, int size);
// extern "C" void gpu_getData(int *src, int *dst, int size);
// extern "C" void gpu_addVectors(int *a_d, int *b_d, int *c_d, int size);
// extern "C" void gpu_memRelease(int *a_d, int *b_d, int *c_d);


#define DOMAIN_CELLS (13)

inline int DOMAIN_WRAP(size_t A, bool isPowTwo)
{
	// printf("PowTwo: %i, notPowTwo: %i\n", isPowTwo * (A & (DOMAIN_CELLS-1)), (!isPowTwo) * (A % DOMAIN_CELLS) );
	printf("!isPowTwo: %i, is not Power of two: %i\n", (int)!isPowTwo, (int)!isPowTwo*(A % DOMAIN_CELLS));
	printf("isPowTwo: %i, is Power of two: %i\n", (int)isPowTwo, (int)isPowTwo*(A & (DOMAIN_CELLS-1)));
	printf("return: %i\n", (isPowTwo * (A & (DOMAIN_CELLS-1)) + (!isPowTwo) * (A % DOMAIN_CELLS)) );
	return ( isPowTwo * (A & (DOMAIN_CELLS-1)) + (!isPowTwo) * (A % DOMAIN_CELLS) );
}

bool isDomainPowerOfTwo(size_t x)
{
    return ( (x == (1<<0)) || (x == (1<<1)) || (x == (1<<2)) || (x == (1<<3)) || (x == (1<<4)) ||
             (x == (1<<5)) || (x == (1<<6)) || (x == (1<<7)) || (x == (1<<8)) || (x == (1<<9)) || 
             (x == (1<<10)) || (x == (1<<11)) || (x == (1<<12)) || (x == (1<<13)) || (x == (1<<14)) ||
             (x == (1<<15)) || (x == (1<<16)) || (x == (1<<17)) || (x == (1<<18)) || (x == (1<<19)) ||
             (x == (1<<20)) || (x == (1<<21)) || (x == (1<<22)) || (x == (1<<23)) || (x == (1<<24)) ||
             (x == (1<<25)) || (x == (1<<26)) || (x == (1<<27)) || (x == (1<<28)) || (x == (1<<29)) || 
             (x == (1<<30)) || (x == (1<<31)) );
}

int main(int argc, char **argv)
{
	int myRank, numProcs;
	const size_t elems = atoi(argv[1]);
	int grid_x = atoi(argv[2]);
	int temp = atoi(argv[3]);
	unsigned int dim = atoi(argv[4]);
	char *kernelName = argv[5];

	// MPI initialization functions
	MPI_Init(&argc, &argv);    /// Start MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    /// get current process id
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);    /// get number of processes

	if (1)//myRank == 0)
	{
		// CPU computation
		CProgram cCompileCuda(kernelName);
		CMain *cpu_class = new CMain(elems);

		cpu_class -> memAlloc();	// alloc input arrays on host (CPU)
		cpu_class -> initialize();	// initialize arrays
		cpu_class -> addVectors();	// CPU computation
		//cpu_class -> argSet();
		//cpu_class -> argPrint( (*cpu_class).arguments );
		cout << "CPU:"<<endl;		// print results
		
		if (myRank == 0)
		{
			cpu_class -> resultPrint();
		}

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
		// size_t *globalSize = new size_t(dim);
		// size_t *localSize = new size_t(3);

		gpu_DAPI_class -> getDevice();
		gpu_DAPI_class -> deviceInfo();

		// compile cuda kernel to ptx file
		// if (myRank == 0)
		// {
		// 	printf("CUDA source file compile only by rank= %i \n", myRank);

		// 	cCompileCuda.createCompileCommand((*gpu_DAPI_class).major);		
		// }
		// MPI_Barrier(MPI_COMM_WORLD);

		if (myRank == 0)
		{
			cCompileCuda.createCompileCommand((*gpu_DAPI_class).major);		
		}
		MPI_Barrier(MPI_COMM_WORLD);

		gpu_DAPI_class -> initCUDA();
		gpu_DAPI_class -> initContextModule();
		gpu_DAPI_class -> initHostData();
		gpu_DAPI_class -> setDeviceMemory();
		gpu_DAPI_class -> setData();
		gpu_DAPI_class -> setAllArguments(temp);
		gpu_DAPI_class -> runKernel2(dim, elems, (*gpu_DAPI_class).kernelArgumentVec );
		gpu_DAPI_class -> finish();
		gpu_DAPI_class -> getData();
		gpu_DAPI_class -> resultPrint(temp);
		gpu_DAPI_class -> releaseDeviceMemory();
		gpu_DAPI_class -> finalizeCUDA();
		printf("free cuda context and memory..\n Alles klar! \n");
		
		delete gpu_DAPI_class;
		delete cpu_class;
		// delete globalSize;
		// delete localSize;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	printf("Rank: %i,\n", myRank);

	MPI_Finalize();

	int A = temp;
	bool isPowOfTwo = isDomainPowerOfTwo(DOMAIN_CELLS);
	int test_tmp = DOMAIN_WRAP(A, isPowOfTwo);
	printf("test_tmp: %i\nisPowOfTwo: %i\nmod(A, DOMAIN_CELLS): %i\nA & (DOMAIN_CELLS-1): %i\n", test_tmp, isPowOfTwo, (A % DOMAIN_CELLS), (A & (DOMAIN_CELLS-1)) );
	
	return 0;
}



// nvcc compile works in this way
/*
nvcc -x cu -arch=sm_20 -I. -dc main.cpp -o main.o 
nvcc -x cu -arch=sm_20 -I. -dc classes.cpp -o classes.o 
nvcc -x cu -arch=sm_20 -I. -dc cuDriverWrapper.hpp -o cuDriverWrapper.o
nvcc -arch=sm_20 main.o classes.o cuDriverWrapper.o -o app

*/