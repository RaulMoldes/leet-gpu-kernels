#include <cuda_runtime.h>


// Aid for unrolling the last loop iterations
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// Kernel for parallel reduction using shared memory
template <int blockSize>
__global__ void reduceSum(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;


    // Each thread loads one element into shmem. Precompute the first sum for more utilization.
    sdata[tid] = (i < n) ? input[i] + input[i + blockSize] : 0.0f;
    __syncthreads();

    // Reduce in steps to minimize sincronization
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);


    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}



// input, output are device pointers
void solve(const float* input, float* output, int N) {
    float* d_input, * d_temp;

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_temp, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configuration parameters
    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    float* current_input = d_input;
    float* current_output = d_temp;
    int current_size = N;

    // Perform reduction in multiple passes if needed
    while (current_size > 1) {
        int currentBlocks = (current_size + blockSize - 1) / blockSize;

        // Launch kernel
        reduceSum<blockSize> << <currentBlocks, blockSize, blockSize * sizeof(float) >> > (
            current_input, current_output, current_size
            );

        // Wait for kernel to complete
        cudaDeviceSynchronize();

        // Swap pointers for next iteration
        float* temp = current_input;
        current_input = current_output;
        current_output = temp;

        current_size = currentBlocks;
    }

    // Copy final result back to host
    cudaMemcpy(output, current_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_temp);
}
