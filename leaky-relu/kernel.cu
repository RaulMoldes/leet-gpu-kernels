#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    long globalIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if (globalIdx < N) {

        float value = input[globalIdx];
        if (value <= 0) {
            value *= 0.01;
        }

        output[globalIdx] = value;

    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel << <blocksPerGrid, threadsPerBlock >> > (input, output, N);
    cudaDeviceSynchronize();
}
