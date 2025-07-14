#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reverse_array(float* input, int N) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        int temp = input[globalIdx];
        __syncthreads();
        input[N - 1 - globalIdx] = temp;
    }


}
// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int shared_memory_size = threadsPerBlock * sizeof(float);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array << <blocksPerGrid, threadsPerBlock, shared_memory_size >> > (input, N);
    cudaDeviceSynchronize();
}
