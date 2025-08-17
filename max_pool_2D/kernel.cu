#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cfloat>


__global__ void maxpool2d_kernel(const float* input, float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding) {

    // Compute global indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H_out * W_out;

    if (idx >= total_elements) return;

    // Decode indices (n, c, h_out, w_out)
    int w_out_pos = idx % W_out;
    int h_out_pos = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    // Compute its position
    int h_start = h_out_pos * stride - padding;
    int w_start = w_out_pos * stride - padding;

    // Each thread loads the kernel to shared memory
    extern __shared__ float shared_input[];

    // Compute thread offset
    int tid = threadIdx.x;
    int shared_offset = tid * kernel_size * kernel_size;

    // Load data from kernel to shmem
    int input_base = n * C * H * W + c * H * W;

    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int h_pos = h_start + kh;
            int w_pos = w_start + kw;

            float val = -FLT_MAX;  // Default value for padding

            // Check if we are within the limits
            if (h_pos >= 0 && h_pos < H && w_pos >= 0 && w_pos < W) {
                int input_idx = input_base + h_pos * W + w_pos;
                val = input[input_idx];
            }

            // Load to shmem
            int shared_idx = shared_offset + kh * kernel_size + kw;
            shared_input[shared_idx] = val;
        }
    }

    // Sincronizae this block
    __syncthreads();

    // Find the maximum in shmem
    float max_val = -FLT_MAX;

    for (int i = 0; i < kernel_size * kernel_size; i++) {
        int shared_idx = shared_offset + i;
        max_val = fmaxf(max_val, shared_input[shared_idx]);
    }

    // Write result back to HBM
    output[idx] = max_val;
}


extern "C" void solve(const float* input, float* output,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding) {

    // Calcular dimensiones de salida
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // Calcular número total de elementos de salida
    int output_size = N * C * H_out * W_out;

    // Configurar grid y bloques
    int threads_per_block = 256;
    int blocks = (output_size + threads_per_block - 1) / threads_per_block;

    // Calcular memoria compartida necesaria
    // Cada thread necesita kernel_size * kernel_size elementos
    size_t shared_mem_size = threads_per_block * kernel_size * kernel_size * sizeof(float);

    // Ejecutar kernel con memoria compartida
    maxpool2d_kernel << <blocks, threads_per_block, shared_mem_size >> > (
        input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
}
