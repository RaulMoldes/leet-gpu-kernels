#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void reduceSum(float* input, float* output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory with bounds checking
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    // Check for valid inputs
    if (y_samples == nullptr || result == nullptr || n_samples <= 0) {
        std::cerr << "Error: Invalid input parameters" << std::endl;
        return;
    }

    // Calculate width
    float width = b - a;

    // GPU memory pointers
    float* d_y_samples = nullptr;
    float* d_partial_sums = nullptr;

    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (n_samples + blockSize - 1) / blockSize;

    try {
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_y_samples, n_samples * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(float)));

        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpy(d_y_samples, y_samples, n_samples * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel for reduction
        int sharedMemSize = blockSize * sizeof(float);
        reduceSum << <numBlocks, blockSize, sharedMemSize >> > (d_y_samples, d_partial_sums, n_samples);

        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy partial sums back to host and sum them
        float* h_partial_sums = new float[numBlocks];
        CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

        float total_sum = 0.0f;
        for (int i = 0; i < numBlocks; i++) {
            total_sum += h_partial_sums[i];
        }

        // Calculate Monte Carlo estimate: average * width
        *result = (total_sum / n_samples) * width;

        // Cleanup host memory
        delete[] h_partial_sums;

    }
    catch (const std::exception& e) {
        std::cerr << "Exception in solve(): " << e.what() << std::endl;
    }

    // Cleanup GPU memory
    if (d_y_samples) cudaFree(d_y_samples);
    if (d_partial_sums) cudaFree(d_partial_sums);
}
