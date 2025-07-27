#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BITS 8
#define RADIX (1 << BITS)
#define MASK (RADIX - 1)
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void histogramKernel(const unsigned int* input, unsigned int* histogramsPerBlock, int n, int shift) {
    __shared__ unsigned int localHist[RADIX];
    int tid = threadIdx.x;

    for (int i = tid; i < RADIX; i += blockDim.x)
        localHist[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        unsigned int bin = (input[idx] >> shift) & MASK;
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    for (int i = tid; i < RADIX; i += blockDim.x) {
        histogramsPerBlock[blockIdx.x * RADIX + i] = localHist[i];
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile unsigned int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <int blockSize>
__global__ void reduceHistograms(const unsigned int* __restrict__ input, unsigned int* __restrict__ output, int numHistograms) {
    extern __shared__ unsigned int sdata[];
    int bin = blockIdx.x;
    int tid = threadIdx.x;
    int offset = bin;

    unsigned int sum = 0;
    for (int i = tid; i < numHistograms; i += blockSize * 2) {
        unsigned int idx1 = i * RADIX + offset;
        unsigned int idx2 = (i + blockSize < numHistograms) ? (i + blockSize) * RADIX + offset : idx1;
        unsigned int val1 = (i < numHistograms) ? input[idx1] : 0;
        unsigned int val2 = (i + blockSize < numHistograms) ? input[idx2] : 0;
        sum += val1 + val2;
    }

    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    if (tid == 0) output[bin] = sdata[0];
}

__global__ void exclusiveScanKernel(unsigned int* data, unsigned int* output, int n) {
    __shared__ unsigned int temp[RADIX * 2];
    int tid = threadIdx.x;

    temp[tid] = (tid < n) ? data[tid] : 0;

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) temp[n - 1] = 0;

    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    if (tid < n) output[tid] = temp[tid];
}

void computeBlockOffsetsCPU(unsigned int* h_histogramsPerBlock, unsigned int* h_blockOffsets, int numBlocks) {
    for (int bin = 0; bin < RADIX; bin++) {
        unsigned int sum = 0;
        for (int block = 0; block < numBlocks; block++) {
            int idx = block * RADIX + bin;
            h_blockOffsets[idx] = sum;
            sum += h_histogramsPerBlock[idx];
        }
    }
}

__global__ void reorderKernel(const unsigned int* input, unsigned int* output,
    const unsigned int* blockOffsets, const unsigned int* prefixSum,
    int n, int shift, int numBlocks) {
    extern __shared__ unsigned int localOffsets[];
    int tid = threadIdx.x;

    if (tid < RADIX) {
        localOffsets[tid] = prefixSum[tid] + blockOffsets[blockIdx.x * RADIX + tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        unsigned int bin = (input[idx] >> shift) & MASK;
        unsigned int pos = atomicAdd(&localOffsets[bin], 1);
        output[pos] = input[idx];
    }
}

void radixSortStep(unsigned int* d_input, unsigned int* d_output, int n, int shift, int numBlocks) {
    unsigned int* d_histogramsPerBlock, * d_histogram, * d_prefixSum;

    CUDA_CHECK(cudaMalloc(&d_histogramsPerBlock, numBlocks * RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_prefixSum, RADIX * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset(d_histogramsPerBlock, 0, numBlocks * RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_prefixSum, 0, RADIX * sizeof(unsigned int)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    histogramKernel << <numBlocks, BLOCK_SIZE, 0, stream >> > (d_input, d_histogramsPerBlock, n, shift);

    int sharedMemSize = BLOCK_SIZE * sizeof(unsigned int);
    reduceHistograms<BLOCK_SIZE> << <RADIX, BLOCK_SIZE, sharedMemSize, stream >> > (
        d_histogramsPerBlock, d_histogram, numBlocks);

    exclusiveScanKernel << <1, RADIX, 0, stream >> > (d_histogram, d_prefixSum, RADIX);

    unsigned int* h_histogramsPerBlock, * h_blockOffsets;
    CUDA_CHECK(cudaHostAlloc(&h_histogramsPerBlock, numBlocks * RADIX * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_blockOffsets, numBlocks * RADIX * sizeof(unsigned int), cudaHostAllocDefault));

    CUDA_CHECK(cudaMemcpyAsync(h_histogramsPerBlock, d_histogramsPerBlock,
        numBlocks * RADIX * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    computeBlockOffsetsCPU(h_histogramsPerBlock, h_blockOffsets, numBlocks);

    unsigned int* d_blockOffsets;
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * RADIX * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyAsync(d_blockOffsets, h_blockOffsets,
        numBlocks * RADIX * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

    reorderKernel << <numBlocks, BLOCK_SIZE, RADIX * sizeof(unsigned int), stream >> > (
        d_input, d_output, d_blockOffsets, d_prefixSum, n, shift, numBlocks);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_blockOffsets);
    cudaFree(d_histogramsPerBlock);
    cudaFree(d_histogram);
    cudaFree(d_prefixSum);
    cudaFreeHost(h_histogramsPerBlock);
    cudaFreeHost(h_blockOffsets);
    cudaStreamDestroy(stream);
}

extern "C" void solve(unsigned int* input, unsigned int* output, int N) {
    unsigned int* d_currInput = input;
    unsigned int* d_currOutput = output;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int shift = 0; shift < 32; shift += BITS) {
        radixSortStep(d_currInput, d_currOutput, N, shift, numBlocks);
        std::swap(d_currInput, d_currOutput);
    }

    if (d_currInput != output) {
        CUDA_CHECK(cudaMemcpy(output, d_currInput, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
}
