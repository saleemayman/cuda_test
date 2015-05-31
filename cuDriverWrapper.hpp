// CUDA driver API wrapper for vector addition
#ifndef __DRIVERAPI_H__
#define __DRIVERAPI_H__

//#include "vecAdd.h"
#include <iostream>
#include <stdio.h>
#include <cmath>
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
    CUdevprop   deviceProps;
    CUdevice    device;
    CUcontext   context;
    CUmodule    module;
    CUfunction  kernel;
    CUstream    stream;

    CUdeviceptr a_d, b_d, c_d;
    CUresult err;

    int deviceCount;
    size_t size;
    float *a_h, *b_h, *c_h; // host pointers
    std::string module_file, kernel_name;

//  char *module_file = (char*) "vecAdd.ptx";
//  char *kernel_name = (char*) "vecAdd";

public:
    int major, minor, driverVer, devAttr;
    std::vector<void *> kernelArgumentVec;

    //constructor
    DriverAPI(size_t _size)
    {
        size = _size;
        // module_file = "vecAdd_s.ptx";
        // kernel_name = "vecAddShared";
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
        cuDeviceGetProperties(&deviceProps, device);

        // printf("Device Information:\n Name: %s \n CC: %d <-> %d \n Driver: v.%d \n", deviceName, major, minor, driverVer/1000);
        // std::cout<<" Compute mode: " << devAttr << std::endl;
        // std::cout << "Max. Shared Mem. per block[bytes]: " << deviceProps.sharedMemPerBlock << std::endl;
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
        err = cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
        // cout << "context: " << err << endl;
        if (err != CUDA_SUCCESS)
        {
            fprintf(stderr, "* Error creating the CUDA context.\n");
            checkCudaErrors( err);
            checkCudaErrors( cuCtxDestroy(context) );
            exit(-1);
        }
    }

    void attachModule(const std::string &file)  //const char &module_file
    {
        moduleMember( file.c_str() );
    }

    void moduleMember(const char *file)
    {
        err = cuModuleLoad(&module, file);  //  err = cuModuleLoad(&module, "vecAdd.ptx");
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
        err = cuModuleGetFunction(&kernel, module, file);   //err = cuModuleGetFunction(&vecAdd, module, "vecAdd");
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
        // c_h = new float[size * 2];

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
        // checkCudaErrors( cuMemAlloc(&c_d, sizeof(float) * size * 2) );
    }

    void setData()
    {
        // copy arrays to device
        checkCudaErrors( cuMemcpyHtoD(a_d, a_h, sizeof(float) * size) );
        checkCudaErrors( cuMemcpyHtoD(b_d, b_h, sizeof(float) * size) );
        checkCudaErrors( cuMemcpyHtoD(c_d, c_h, sizeof(float) * size) );
        for (int i = 0; i < size; i++)
        {
            c_h[i] = -1.0f;
        }
        // checkCudaErrors( cuMemcpyHtoD(c_d, c_h, sizeof(float) * size * 2) );
    }

    void getData()
    {
        // copy array c_d to host array c_h
        checkCudaErrors( cuMemcpyDtoH(c_h, c_d, sizeof(float) * size) );
        // checkCudaErrors( cuMemcpyDtoH(c_h, c_d, sizeof(float) * size * 2) );
    }

    void resultPrint(int temp)
    {
        printf("GPU c_h: \n");
        for(size_t i = 0; i < size; i++)
        {
            printf("\t%i ", (int)c_h[i]);
        }
        printf("\n");
        // printf("c[i + size]: \n");
        // for(size_t i = 0; i < size; i++)
        // {
        //     printf("%lu: \t %i + %i + %i = %i \n", i, (int)a_h[i], (int)b_h[i], temp, (int)c_h[i + size]);
        // }
        // printf("c[%lu]: %f\n", (size - 1), c_h[size - 1]);
//      printf("\n");
    }

    void finalizeCUDA()
    {
        // de allocate host memory
        delete[] a_h;
        delete[] b_h;
        delete[] c_h;

        // printf("\t Host arrays de-allocated\n");
        checkCudaErrors( cuCtxDestroy(context) );
        // printf("\t Context dettached\n");
/*      checkCudaErrors( cuStreamDestroy(stream) );
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
        kernelArgumentVec[index] = &arg;
    }

    inline void setKernelArg(int index, char &arg)
    {
        kernelArgumentVec[index] = &arg;
    }

    inline void setKernelArg(int index, int &arg)
    {
        kernelArgumentVec[index] = &arg;
    }

    inline void setKernelArg(int index, const int &arg)
    {
        kernelArgumentVec[index] = const_cast<int *>(&arg); //const_cast< new_type > ( expression )
    }

    inline void setKernelArg(int index, float &arg)
    {
        kernelArgumentVec[index] = &arg;
    }
    inline void setKernelArg(int index, size_t &arg)
    {
        kernelArgumentVec[index] = &arg;
    }

    inline void finish()
    {
        checkCudaErrors( cuStreamSynchronize(stream) );
    }

    void setAllArguments(int &temp)
    {
        //int temp = -100;

        // setup the arguments for the cuda kernel
        kernelArgumentVec.reserve(10);

        setKernelArg(0, temp);
        setKernelArg(1, a_d);
        setKernelArg(2, b_d);
        setKernelArg(3, c_d);
        setKernelArg(4, size);

//      printf("--args: [0] [1] [2] [3] [4] %i  %i  %i  %i  %i \n", kernelArgumentVec[0], kernelArgumentVec[1], kernelArgumentVec[2], kernelArgumentVec[3], kernelArgumentVec[4]);
    }

    void setGridAndBlockSize( dim3 &numBlocks, dim3 &threadsPerBlock, unsigned int work_dim, const size_t total_elems )
    {
        size_t threads_per_block;

        threadsPerBlock = dim3(LOCAL_WORK_GROUP_SIZE/32, LOCAL_WORK_GROUP_SIZE/32, 1);
        threads_per_block = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

        if (work_dim == 1)
        {
            size_t grid_x = (total_elems + threadsPerBlock.x - 1)/threads_per_block;
            numBlocks = dim3(grid_x + 1, 1, 1);
        }
        else if(work_dim == 2)
        {
            size_t grid_xy = sqrt((total_elems + threadsPerBlock.x * threadsPerBlock.y - 1)/threads_per_block);
            numBlocks = dim3(grid_xy + 1, grid_xy + 1, 1);
        }
        else
        {
            size_t grid_xyz = pow((total_elems + threads_per_block - 1)/threads_per_block, (1/3.));
            numBlocks = dim3(grid_xyz + 1, grid_xyz + 1, grid_xyz + 1);
        }

        // printf("setGridAndBlockSize: numBlocks: [%d %d %d], threadsPerBlock: [%d %d %d] \n", 
        //     numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    }

    void runKernel2(unsigned int work_dim, const size_t global_work_size, std::vector<void *>& arguments)
    {
        dim3 threadsPerBlock, numBlocks;//  = dim3(32, 1, 1);
        size_t sharedMemory;

        setGridAndBlockSize(numBlocks, threadsPerBlock, work_dim, global_work_size);
        // printf("numBlocks: [%d %d %d], threadsPerBlock: [%d %d %d] \n", numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

        sharedMemory = (sizeof(float) * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z) < deviceProps.sharedMemPerBlock?  
                                        sizeof(float) * (threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z) :
                                        deviceProps.sharedMemPerBlock;

        // std::cout << "Block memory size reqd.[bytes]: " << (sizeof(float) * threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z) << std::endl;
        std::cout << std::endl;

        // launch the kernel
        checkCudaErrors( cuLaunchKernel(kernel,
                                        numBlocks.x,
                                        numBlocks.y,
                                        numBlocks.z,  // Nx1x1 blocks
                                        threadsPerBlock.x,
                                        threadsPerBlock.y,
                                        threadsPerBlock.z, // 1x1x1 threads
                                        sharedMemory,
                                        stream,
                                        &arguments[0],
                                        0) );
    }
};

#endif  // __DRIVERAPI_H__