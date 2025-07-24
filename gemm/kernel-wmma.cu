#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// CUDA Kernel: one thread block computes one tile (16x16)
__global__ void gemm_wmma(const __half* A, const __half* B, __half* C,
    float alpha, float beta, int M, int N, int K) {
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Calculate global position
    int c_row = tile_row * WMMA_M;
    int c_col = tile_col * WMMA_N;

    // Check if this tile is within bounds
    if (c_row >= M || c_col >= N) return;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Compute matrix multiplication A * B
    for (int tile_k = 0; tile_k < (K + WMMA_K - 1) / WMMA_K; ++tile_k) {
        int a_col = tile_k * WMMA_K;
        int b_row = tile_k * WMMA_K;

        // Check if we're within K bounds
        if (a_col >= K) break;

        // Create padded tiles for A and B if necessary
        __shared__ __half shared_A[WMMA_M * WMMA_K];
        __shared__ __half shared_B[WMMA_K * WMMA_N];

        // Load A tile with padding
        for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < WMMA_K; j += blockDim.x) {
                int global_row = c_row + i;
                int global_col = a_col + j;

                if (global_row < M && global_col < K) {
                    shared_A[i * WMMA_K + j] = A[global_row * K + global_col];
                }
                else {
                    shared_A[i * WMMA_K + j] = __float2half(0.0f);
                }
            }
        }

        // Load B tile with padding
        for (int i = threadIdx.y; i < WMMA_K; i += blockDim.y) {
            for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
                int global_row = b_row + i;
                int global_col = c_col + j;

                if (global_row < K && global_col < N) {
                    shared_B[i * WMMA_N + j] = B[global_row * N + global_col];
                }
                else {
                    shared_B[i * WMMA_N + j] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // Load fragments from shared memory
        wmma::load_matrix_sync(a_frag, shared_A, WMMA_K);
        wmma::load_matrix_sync(b_frag, shared_B, WMMA_N);

        // Perform matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // Handle alpha and beta scaling
    if (beta != 0.0f) {
        // Load the original C matrix
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        // Create a padded C tile in shared memory
        __shared__ float shared_C[WMMA_M * WMMA_N];

        // Load C values with padding
        for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
            for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
                int global_row = c_row + i;
                int global_col = c_col + j;

                if (global_row < M && global_col < N) {
                    shared_C[i * WMMA_N + j] = __half2float(C[global_row * N + global_col]);
                }
                else {
                    shared_C[i * WMMA_N + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Load C fragment
        wmma::load_matrix_sync(c_frag, shared_C, WMMA_N, wmma::mem_row_major);

        // Compute alpha * acc + beta * c using element-wise operations
        for (int i = 0; i < c_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    }
    else {
        // Just scale by alpha
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }

    // Store the result back to C
    __shared__ float shared_result[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(shared_result, acc_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    // Copy from shared memory to global memory (only within bounds)
    for (int i = threadIdx.y; i < WMMA_M; i += blockDim.y) {
        for (int j = threadIdx.x; j < WMMA_N; j += blockDim.x) {
            int global_row = c_row + i;
            int global_col = c_col + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = __float2half(shared_result[i * WMMA_N + j]);
            }
        }
    }
}

extern "C" void solve(const __half* A, const __half* B, __half* C, float alpha, float beta,
    int M, int N, int K) {
    // GPU pointers
    __half* d_A, * d_B, * d_C;

    size_t sizeA = M * K * sizeof(__half);
    size_t sizeB = K * N * sizeof(__half);
    size_t sizeC = M * N * sizeof(__half);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    dim3 blockDim(32, 4); // 128 threads per block
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    // Launch kernel
    gemm_wmma << <gridDim, blockDim >> > (d_A, d_B, d_C, alpha, beta, M, N, K);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy((void*)C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
