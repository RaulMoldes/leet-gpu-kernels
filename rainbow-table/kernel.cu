#include <iostream>
#include <vector>
#include <cuda_runtime.h>


__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;

    unsigned int hash = OFFSET_BASIS;

    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }

    return hash;
}




// Versión optimizada usando shared memory y tiles
__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    // Shared memory para tile de datos
    __shared__ unsigned int shared_data[256];

    int tid = threadIdx.x;  // ID local del hilo
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Verificar límites
    if (global_idx >= N) return;

    // PASO 1: Carga cooperativa a shared memory
    // Todos los hilos del bloque cargan datos juntos
    shared_data[tid] = (unsigned int)input[global_idx];
    __syncthreads();  // Sincronizar carga

    // PASO 2: Procesamiento en shared memory (más rápida)
    unsigned int current_value = shared_data[tid];

    // PASO 3: Aplicar hash R veces usando datos en shared memory
    for (int round = 0; round < R; round++) {
        current_value = fnv1a_hash((int)current_value);
    }

    // PASO 4: Escritura coalescente a memoria global
    output[global_idx] = current_value;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel << <blocksPerGrid, threadsPerBlock >> > (input, output, N, R);
    cudaDeviceSynchronize();
}
