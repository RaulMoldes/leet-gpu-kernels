#include <cuda_runtime.h>
#define TILE_SIZE 16
#define BLOCK_SIZE 256

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = block_row + threadIdx.y;
    int thread_col = block_col + threadIdx.x;

    if (thread_row < rows && thread_col < cols) {
        int input_idx = thread_row * cols + thread_col;
        tile[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    else {
        tile[threadIdx.y][threadIdx.x] = 0.0f; // padding for shared memory
    }

    __syncthreads();

    int output_row = block_col + threadIdx.y;
    int output_col = block_row + threadIdx.x;

    if (output_row < cols && output_col < rows) {
        int output_idx = output_row * rows + output_col;
        output[output_idx] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
        (rows + TILE_SIZE - 1) / TILE_SIZE);

    matrix_transpose_kernel << <blocksPerGrid, threadsPerBlock >> > (input, output, rows, cols);
    cudaDeviceSynchronize();
}
