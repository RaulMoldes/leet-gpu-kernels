#include <cuda_runtime.h>
#include <cstdio>


// Kernel for performing matmul in COMPRESSED SPARSE ROW FORMAT
// CSR (Compressed Sparse Row) represents:
// values   = [10, 2, 3, 4]        // Non zero values in the matrix
// col_idx = [0, 2, 0, 2]         // The column of each value
// row_ptr = [0, 1, 2, 4]         // Indicates where each row starts at values and col_idx.
__global__ void spmv_csr_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* values,
    const float* x,
    float* y,
    int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; ++i) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

extern "C" void solve(
    const int* h_row_ptr,
    const int* h_col_idx,
    const float* h_values,
    const float* h_x,
    float* h_y,
    int M,
    int N,
    int nnz)
{
    // Device pointers
    int* d_row_ptr, * d_col_idx;
    float* d_values, * d_x, * d_y;

    // Allocate device memory
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    spmv_csr_kernel << <blocks, threads >> > (d_row_ptr, d_col_idx, d_values, d_x, d_y, M);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
