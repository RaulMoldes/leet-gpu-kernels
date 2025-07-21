#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

// Kernel para prefix sum por bloque
__global__ void scan_kernel(const float* input, float* output, float* block_sums, int N) {
    __shared__ float temp[2 * BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Cargar datos
    if (gid < N) {
        temp[tid] = input[gid];
    }
    else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Inclusive scan (Hillis-Steele)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float t = 0.0f;
        if (tid >= offset) t = temp[tid - offset];
        __syncthreads();
        temp[tid] += t;
        __syncthreads();
    }

    // Escribir resultado
    if (gid < N) {
        output[gid] = temp[tid];
    }

    // Guardar el total del bloque
    if (block_sums && tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[tid];
    }
}

// Kernel para aplicar offset por bloque
__global__ void add_offsets(float* output, const float* block_sums_scan, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (blockIdx.x == 0 || gid >= N) return;

    float offset = block_sums_scan[blockIdx.x - 1];
    output[gid] += offset;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {

    float* d_input, * d_output, * d_block_sums, * d_block_sums_scan;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    cudaMalloc(&d_block_sums_scan, num_blocks * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Paso 1: prefix sum por bloque
    scan_kernel << <num_blocks, BLOCK_SIZE >> > (d_input, d_output, d_block_sums, N);
    cudaDeviceSynchronize();

    // Paso 2: prefix sum de los bloques
    if (num_blocks > 1) {
        scan_kernel << <1, BLOCK_SIZE >> > (d_block_sums, d_block_sums_scan, nullptr, num_blocks);
        cudaDeviceSynchronize();

        // Paso 3: aplicar offset a cada bloque
        add_offsets << <num_blocks, BLOCK_SIZE >> > (d_output, d_block_sums_scan, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scan);
}
