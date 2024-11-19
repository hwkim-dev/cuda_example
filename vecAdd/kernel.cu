
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>

cudaError_t vecAdd(float* A_h, float* B_h, float* C_h, int n);

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int n = 10;

    float a[n] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1};
    float b[n] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1};
    float c[n] = {};
    cudaError_t cudaStatus = vecAdd(a, b, c, n);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cudaDeviceReset();
    
    int i = 0;
    do 
    {
        printf("%.1f, ", c[i]);
        ++i;
    } while (i < n - 1);
    
    printf("%.1f", c[n-1]);

    return 0;
}



cudaError_t vecAdd(float* A_h, float* B_h, float* C_h, int n)
{
    int size = n * sizeof(float);
    float *A_d, *B_d, * C_d;

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    //Allocate device memory for A, B, C
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    //copy memory(host=A_h, B_h) to memory(device=A_d, B_d)
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    //<<<dimGrid, dimBlock>>>
    vecAddKernel<<< ceil(n / 256.0), 256 >>>(A_d, B_d, C_d, n);
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }

    //copy result(memory) from Device to Host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return cudaStatus;
}

