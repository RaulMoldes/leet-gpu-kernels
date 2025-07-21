#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>

// THIS ONLY PERFORMS NORMAL ATTENTION. MY GOAL IS TO IMPLEMENT FLASH ATTENTION IN THE FUTURE.

// Q: (M x D), K: (N x D) => scores: (M x N)
__global__ void computeScoresKernel(float* Q, float* K, float* scores, int M, int N, int D) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float score = 0.0f;
        for (int d = 0; d < D; ++d)
            score += Q[m * D + d] * K[n * D + d];

        scores[m * N + n] = score / sqrtf((float)D);
    }
}

// Softmax por fila (por M)
__global__ void applySoftmaxKernel(float* scores, float* softmax, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float maxVal = -1e30f;
    for (int n = 0; n < N; ++n)
        maxVal = fmaxf(maxVal, scores[m * N + n]);

    float sumExp = 0.0f;
    for (int n = 0; n < N; ++n) {
        softmax[m * N + n] = expf(scores[m * N + n] - maxVal);
        sumExp += softmax[m * N + n];
    }

    for (int n = 0; n < N; ++n)
        softmax[m * N + n] /= sumExp;
}

// Output: softmax (M x N) * V (N x D) = Output (M x D)
__global__ void computeOutputKernel(float* softmax, float* V, float* output, int M, int N, int D) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && d < D) {
        float val = 0.0f;
        for (int n = 0; n < N; ++n)
            val += softmax[m * N + n] * V[n * D + d];

        output[m * D + d] = val;
    }
}

// Complete solve
extern "C" void solve(float* Q, float* K, float* V, float* output, int M, int N, int D) {
    float* d_Q, * d_K, * d_V, * d_scores, * d_softmax, * d_output;

    cudaMalloc(&d_Q, M * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_scores, M * N * sizeof(float));
    cudaMalloc(&d_softmax, M * N * sizeof(float));
    cudaMalloc(&d_output, M * D * sizeof(float));

    cudaMemcpy(d_Q, Q, M * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: QK^T
    dim3 block1(16, 16);
    dim3 grid1((N + 15) / 16, (M + 15) / 16);
    computeScoresKernel << <grid1, block1 >> > (d_Q, d_K, d_scores, M, N, D);
    cudaDeviceSynchronize();

    // Step 2: softmax
    int block2 = 128;
    int grid2 = (M + block2 - 1) / block2;
    applySoftmaxKernel << <grid2, block2 >> > (d_scores, d_softmax, M, N);
    cudaDeviceSynchronize();

    // Step 3: softmax * V
    dim3 block3(16, 16);
    dim3 grid3((D + 15) / 16, (M + 15) / 16);
    computeOutputKernel << <grid3, block3 >> > (d_softmax, d_V, d_output, M, N, D);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, M * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_softmax);
    cudaFree(d_output);
}
