#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <vector>


// =============================================================================
// WARP REDUCTION PRIMITIVES
// =============================================================================

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

// =============================================================================
// BLOCK REDUCTION PRIMITIVES
// =============================================================================

template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
    __shared__ T shared[33];  // +1 para evitar bank conflicts

    int lane = threadIdx.x & 0x1f;  // threadIdx.x % 32
    int wid = threadIdx.x >> 5;     // threadIdx.x / 32

    // Primero reduce dentro del warp
    val = warpReduceMax(val);

    // Thread 0 de cada warp guarda en shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Primer warp hace la reducción final
    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : -FLT_MAX;
        val = warpReduceMax(val);
    }

    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    __shared__ T shared[33];

    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    return val;
}

// =============================================================================
// SOFTMAX KERNEL
// =============================================================================

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Usar shared memory para cachear datos
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // PASO 1: Encontrar el máximo global usando block reduction
    float local_max = -FLT_MAX;

    // Cada thread encuentra su máximo local
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_max = max(local_max, input[i]);
    }

    // Reducir máximo entre todos los threads del bloque
    float block_max = blockReduceMax(local_max);

    // Broadcast del máximo a todos los threads
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = block_max;
    }
    __syncthreads();

    // PASO 2: Calcular suma de exponenciales
    float local_sum = 0.0f;

    // Cada thread calcula su suma local
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += expf(input[i] - s_max);
    }

    // Reducir suma entre todos los threads del bloque
    float block_sum = blockReduceSum(local_sum);

    // Broadcast de la suma a todos los threads
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = block_sum;
    }
    __syncthreads();

    // PASO 3: Calcular softmax final y escribir resultado
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = expf(input[i] - s_max) / s_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Limitar número de bloques para el caso de N grande
    blocksPerGrid = min(blocksPerGrid, 65535);

    // Lanzar kernel con shared memory
    size_t shared_mem = 0;  // No necesitamos shared memory extra en esta versión

    softmax_kernel << <blocksPerGrid, threadsPerBlock, shared_mem >> > (input, output, N);
    cudaDeviceSynchronize();
}
