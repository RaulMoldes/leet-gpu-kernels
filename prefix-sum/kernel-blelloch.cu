#include <cuda_runtime.h>
#define BLOCK_SIZE 512

// Kernel para prefix sum por bloque usando Blelloch scan (work-efficient)
__global__ void scan_kernel(const float* input, float* output, float* block_sums, int N) {
    __shared__ float temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Cargar datos en shared memory
    if (gid < N) {
        temp[tid] = input[gid];
    }
    else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    int offset = 1;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (ai < blockDim.x && bi < blockDim.x) {
                temp[bi] += temp[ai];
            }
        }
        offset <<= 1;
    }

    // Clear the last element
    if (tid == 0) {
        if (block_sums) {
            block_sums[blockIdx.x] = temp[blockDim.x - 1];
        }
        temp[blockDim.x - 1] = 0.0f;
    }

    // Down-sweep phase
    for (int d = 1; d < blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            if (ai < blockDim.x && bi < blockDim.x) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    __syncthreads();

    // Escribir resultado (exclusive scan convertido a inclusive)
    if (gid < N) {
        if (gid == 0) {
            output[gid] = input[gid];  // Primer elemento para inclusive scan
        }
        else {
            output[gid] = temp[tid] + input[gid];
        }
    }
}

// Kernel para aplicar offset por bloque
__global__ void add_offsets(float* output, const float* block_sums_scan, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 || gid >= N) return;

    output[gid] += block_sums_scan[blockIdx.x - 1];
}

// Función recursiva para manejar múltiples niveles
void prefix_sum_recursive(float* d_input, float* d_output, int N) {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (num_blocks == 1) {
        // Caso base: un solo bloque
        scan_kernel << <1, BLOCK_SIZE >> > (d_input, d_output, nullptr, N);
        return;
    }

    // Paso 1: Scan por bloques
    float* d_block_sums, * d_block_sums_scan;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    cudaMalloc(&d_block_sums_scan, num_blocks * sizeof(float));

    scan_kernel << <num_blocks, BLOCK_SIZE >> > (d_input, d_output, d_block_sums, N);
    cudaDeviceSynchronize();

    // Paso 2: Scan recursivo de las sumas de bloques
    prefix_sum_recursive(d_block_sums, d_block_sums_scan, num_blocks);

    // Paso 3: Aplicar offsets
    add_offsets << <num_blocks, BLOCK_SIZE >> > (d_output, d_block_sums_scan, N);
    cudaDeviceSynchronize();

    cudaFree(d_block_sums);
    cudaFree(d_block_sums_scan);
}

// Función principal
extern "C" void solve(const float* input, float* output, int N) {
    float* d_input, * d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    prefix_sum_recursive(d_input, d_output, N);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
