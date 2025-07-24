#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>

__device__ float log_sum_exp(const float* logits, int C) {
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < C; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    // Compute sum of exp(logits - max_logit)
    float sum_exp = 0.0f;
    for (int i = 0; i < C; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return max_logit + logf(sum_exp);
}

__global__ void cross_entropy_kernel(const float* logits, const int* true_labels, int N, int C, float* losses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* sample_logits = logits + idx * C;
    int true_label = true_labels[idx];

    float log_sum = log_sum_exp(sample_logits, C);
    float loss = -sample_logits[true_label] + log_sum;

    losses[idx] = loss;
}

// Parallel reduction kernel for summing losses (block-level reduction)
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_sum(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float x = (idx < N) ? input[idx] : 0.0f;
    x = warpReduceSum(x);

    if ((tid & 31) == 0) sdata[tid / 32] = x;
    __syncthreads();

    // Reduce within the first warp
    if (tid < 32) {
        float val = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) output[blockIdx.x] = val;
    }
}

extern "C" void solve(const float* logits, const int* true_labels, int N, int C, float* loss) {
    // Allocate device buffers
    float* d_logits;
    int* d_true_labels;
    float* d_losses;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_true_labels, N * sizeof(int));
    cudaMalloc(&d_losses, N * sizeof(float));

    cudaMemcpy(d_logits, logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true_labels, true_labels, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to compute per-sample losses
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cross_entropy_kernel << <blocks, threads >> > (d_logits, d_true_labels, N, C, d_losses);

    // Reduce sum of losses
    // We'll do multi-step reduction until result fits in one float
    float* d_in = d_losses;
    float* d_out;
    cudaMalloc(&d_out, blocks * sizeof(float));

    int n = N;
    int current_blocks = blocks;

    while (current_blocks > 1) {
        int shared_mem = (threads / 32) * sizeof(float);
        reduce_sum << <current_blocks, threads, shared_mem >> > (d_in, d_out, n);
        n = current_blocks;
        current_blocks = (n + threads - 1) / threads;

        // Swap buffers
        float* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // Final reduction to get the sum of losses
    int shared_mem = (threads / 32) * sizeof(float);
    reduce_sum << <1, threads, shared_mem >> > (d_in, d_out, n);

    // Copy back the sum loss
    float sum_loss;
    cudaMemcpy(&sum_loss, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Average loss
    *loss = sum_loss / N;

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_true_labels);
    cudaFree(d_losses);
    cudaFree(d_out);
}
