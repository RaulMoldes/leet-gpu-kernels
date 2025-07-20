#include <cuda_runtime.h>
#include <cmath>

__global__ void relu_kernel(const float* input, float* output, int N) {
    long globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        output[globalIdx] = fmaxf(0, input[globalIdx]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel << <blocksPerGrid, threadsPerBlock >> > (input, output, N);
    cudaDeviceSynchronize();
}
