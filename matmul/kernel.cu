#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>



#define TILE_SIZE 16
// I have seen some cache access patterns that can lead to cache thrashing
// and performance degradation.
// This happens when the matrix dimensions are exact multiples of cache line
// boundaries and cache set counts, particularly powers of 2 like 2048, 4096.
// When matrix width = 2048 floats = 8192 bytes:
// - Each row starts at address: base + row * 2048 * 4 bytes
// - Cache set mapping: (address >> 7) & set_mask
// - Row stride = 8192 bytes = 64 cache lines * 4 byte per line on most GPUs
// - With 2048 cache sets, every 32nd row maps to same set
// - This causes cache line evictions and thrashing.
// To avoid this, I recommend using matrix dimensions that are ot exact powers of 2.
// If not possible, consider data layout changes or feel confortable wih the performance impact.
__global__ void matmul(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];


    // COLLABORATIVE LOAD COORDINATES (EACH THREAD LOADS ONE ELEMENT INTO SHARED MEM)
    int load_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int load_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // COMPUTE COORDINATES (EACH THREAD COMPUTES A 2X2 BLOCK)
    int row_base = blockIdx.y * TILE_SIZE + threadIdx.y * 2;
    int col_base = blockIdx.x * TILE_SIZE + threadIdx.x * 2;

    // Double tiling (each thread computes a 2x2 mini block)
    float sum[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // COLLABORATIVE LOAD
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (load_row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[load_row * K + a_col];
        }
        else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && load_col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + load_col];
        }
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }


        __syncthreads();
        if (threadIdx.y < TILE_SIZE / 2 && threadIdx.x < TILE_SIZE / 2) {
#pragma unroll
            // Compute 2x2 block
            for (int k = 0; k < TILE_SIZE; ++k) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        int tile_row = threadIdx.y * 2 + i;
                        int tile_col = threadIdx.x * 2 + j;
                        if (tile_row < TILE_SIZE && tile_col < TILE_SIZE) {
                            sum[i][j] += tile_A[tile_row][k] * tile_B[k][tile_col];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int row = row_base + i;
            int col = col_base + j;
            if (row < M && col < N && threadIdx.y < TILE_SIZE / 2 && threadIdx.x < TILE_SIZE / 2) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}


void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
