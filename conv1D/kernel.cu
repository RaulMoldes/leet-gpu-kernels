#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
    int input_size, int kernel_size) {

    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int output_size = input_size - kernel_size + 1;

    // Número de elementos que procesará cada bloque
    int elements_per_block = blockDim.x;
    int block_start = bid * elements_per_block;

    // Cargar kernel a memoria compartida
    if (tid < kernel_size) {
        s_data[tid] = kernel[tid];
    }

    // Calcular cuántos elementos de input necesita este bloque
    int input_needed = elements_per_block + kernel_size - 1;
    int input_start = block_start;

    // Cargar input a memoria compartida (después del kernel)
    float* s_input = &s_data[kernel_size];

    for (int i = tid; i < input_needed; i += blockDim.x) {
        int global_idx = input_start + i;
        if (global_idx < input_size) {
            s_input[i] = input[global_idx];
        }
        else {
            s_input[i] = 0.0f;
        }
    }

    __syncthreads();

    // Cada thread procesa un elemento de salida
    int output_idx = block_start + tid;

    if (output_idx < output_size) {
        float sum = 0.0f;

#pragma unroll 8
        for (int k = 0; k < kernel_size; k++) {
            sum += s_input[tid + k] * s_data[k];
        }

        output[output_idx] = sum;
    }

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int blockSize = 512;
    int gridSize = (output_size + blockSize - 1) / blockSize;

    // Memoria compartida: kernel + input del bloque
    int shared_mem = (kernel_size + blockSize + kernel_size - 1) * sizeof(float);


    // Limitar shared memory a 48KB
    if (shared_mem > 48 * 1024) {
        blockSize = 256;
        gridSize = (output_size + blockSize - 1) / blockSize;
        shared_mem = (kernel_size + blockSize + kernel_size - 1) * sizeof(float);
    }

    convolution_1d_kernel << <gridSize, blockSize, shared_mem >> > (
        input, kernel, output, input_size, kernel_size);
}
