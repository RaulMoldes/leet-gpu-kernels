#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 16

__global__ void shared_compute_scores(float* Q, float* K, float* scores, int M, int N, int D) {
    __shared__ float tile_Q[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_K[TILE_WIDTH][TILE_WIDTH];

    int m = blockIdx.y * TILE_WIDTH + threadIdx.y; // fila Q
    int n = blockIdx.x * TILE_WIDTH + threadIdx.x; // columna K^T (fila de K)

    float score = 0.0f;
    int numPhases = (D + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numPhases; t++) {
        // Q_tile: fila m, columna t*TILE_WIDTH + x
        if (m < M && t * TILE_WIDTH + threadIdx.x < D)
            tile_Q[threadIdx.y][threadIdx.x] = Q[m * D + t * TILE_WIDTH + threadIdx.x];
        else
            tile_Q[threadIdx.y][threadIdx.x] = 0.0f;

        // K_tile: fila n, columna t*TILE_WIDTH + y
        if (n < N && t * TILE_WIDTH + threadIdx.y < D)
            tile_K[threadIdx.y][threadIdx.x] = K[n * D + t * TILE_WIDTH + threadIdx.y];
        else
            tile_K[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
            score += tile_Q[threadIdx.y][i] * tile_K[i][threadIdx.x];

        __syncthreads();
    }

    if (m < M && n < N)
        scores[m * N + n] = score / sqrtf((float)D);
}

__global__ void shared_softmax(float* scores, float* softmax, int M, int N) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    float maxVal = -1e30f;  // Calcular el maximo local
    for (int n = 0; n < N; ++n)
        maxVal = fmaxf(maxVal, scores[m * N + n]);

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        softmax[m * N + n] = expf(scores[m * N + n] - maxVal);
        sum += softmax[m * N + n];
    }

    for (int n = 0; n < N; ++n)
        softmax[m * N + n] /= sum;
}

__global__ void shared_compute_output(float* softmax, float* V, float* output, int M, int N, int D) {
    __shared__ float tile_softmax[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_V[TILE_WIDTH][TILE_WIDTH];

    int m = blockIdx.y * TILE_WIDTH + threadIdx.y; // fila output
    int d = blockIdx.x * TILE_WIDTH + threadIdx.x; // columna output

    float outVal = 0.0f;
    int numPhases = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numPhases; t++) {
        // Load softmax tile (M x N)
        if (m < M && t * TILE_WIDTH + threadIdx.x < N)
            tile_softmax[threadIdx.y][threadIdx.x] = softmax[m * N + t * TILE_WIDTH + threadIdx.x];
        else
            tile_softmax[threadIdx.y][threadIdx.x] = 0.0f;

        // Load V tile (N x D)
        if (t * TILE_WIDTH + threadIdx.y < N && d < D)
            tile_V[threadIdx.y][threadIdx.x] = V[(t * TILE_WIDTH + threadIdx.y) * D + d];
        else
            tile_V[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
            outVal += tile_softmax[threadIdx.y][i] * tile_V[i][threadIdx.x];

        __syncthreads();
    }

    if (m < M && d < D)
        output[m * D + d] = outVal;
}


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

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_scores((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    shared_compute_scores << <grid_scores, block >> > (d_Q, d_K, d_scores, M, N, D);
    cudaDeviceSynchronize();

    int block_softmax = 128;
    int grid_softmax = (M + block_softmax - 1) / block_softmax;
    shared_softmax << <grid_softmax, block_softmax >> > (d_scores, d_softmax, M, N);
    cudaDeviceSynchronize();

    dim3 grid_output((D + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    shared_compute_output << <grid_output, block >> > (d_softmax, d_V, d_output, M, N, D);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, M * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_softmax);
    cudaFree(d_output);
}
