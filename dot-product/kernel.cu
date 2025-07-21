#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

__global__ void dot_product_kernel(const float* A, const float* B, float* partial_sums, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < N) {
        val = A[i] * B[i];
    }

    sdata[tid] = val;
    __syncthreads();

    // Reducción paralela en shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Escribir suma parcial del bloque
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}



extern "C" void solve(const float* A_host, const float* B_host, float* result, int N) {
    float* d_A, * d_B, * d_partial_sums;

    // Número de bloques necesarios
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));

    cudaMemcpy(d_A, A_host, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host, N * sizeof(float), cudaMemcpyHostToDevice);

    // Lanzar kernel
    dot_product_kernel << <num_blocks, BLOCK_SIZE >> > (d_A, d_B, d_partial_sums, N);
    cudaDeviceSynchronize();

    // Traer resultados parciales al host
    float* h_partial_sums = new float[num_blocks];
    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Reducción final en CPU
    float sum = 0.0f;
    for (int i = 0; i < num_blocks; ++i) {
        sum += h_partial_sums[i];
    }

    *result = sum;

    // Limpieza
    delete[] h_partial_sums;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial_sums);
}
