#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

__global__ void histogram_kernel(const int* input, int N, int* histogram, int num_bins) {
    extern __shared__ int shared_bins[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Initialize shared histogram to 0
    for (int i = local_tid; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }

    __syncthreads();

    // Process elements and build local histogram
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        int val = input[i];
        if (val >= 0 && val < num_bins) {
            atomicAdd(&shared_bins[val], 1);
        }
    }

    __syncthreads();

    // Reduce shared histograms into global histogram
    for (int i = local_tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], shared_bins[i]);
    }
}

extern "C" void solve(const int* input, int N, int* histogram, int num_bins) {
    int* d_input, * d_histogram;

    size_t input_size = N * sizeof(int);
    size_t hist_size = num_bins * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_histogram, hist_size);

    // Copy input to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, hist_size);

    // Kernel launch config
    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    blocks = min(blocks, 1024); // Limit max blocks for practicality

    size_t shared_mem_size = num_bins * sizeof(int);
    histogram_kernel << <blocks, threads, shared_mem_size >> > (d_input, N, d_histogram, num_bins);

    // Copy histogram back to host
    cudaMemcpy(histogram, d_histogram, hist_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histogram);
}
