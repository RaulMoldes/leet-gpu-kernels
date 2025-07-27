#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
            exit(EXIT_FAILURE);                                           \
        }                                                                  \
    } while (0)

// Devuelve siguiente potencia de 2
__host__ int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Bitonic sort kernel adaptado para floats
__global__ void bitonicSortKernel(float* data, int size, int j, int k) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = idx ^ j;

    if (ixj > idx && idx < size && ixj < size) {
        if ((idx & k) == 0) {
            if (data[idx] < data[ixj]) {
                float tmp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = tmp;
            }
        }
        else {
            if (data[idx] > data[ixj]) {
                float tmp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}

// Función de ordenamiento + top-k
extern "C" void solve(const float* input, float* output, int N, int k) {
    int n = nextPowerOfTwo(N);
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, input, N * sizeof(float), cudaMemcpyDeviceToDevice));

    // Rellenar con +inf si es necesario
    if (n > N) {
        float inf = INFINITY;
        for (int i = N; i < n; ++i) {
            CUDA_CHECK(cudaMemcpy(d_data + i, &inf, sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            bitonicSortKernel << <blocks, threads >> > (d_data, n, stride, size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Copiar los primeros k al output
    CUDA_CHECK(cudaMemcpy(output, d_data, k * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_data));
}

// Main para probar con array grande
int main() {
    const int N = 1 << 20;  // 1 millón de elementos
    const int K = 10;

    std::vector<float> h_input(N);
    std::vector<float> h_output(K);

    // Rellenar con números aleatorios
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(std::rand()) / RAND_MAX * 1000.0f;
    }

    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Ordenar y tomar top-k
    solve(d_input, d_output, N, K);

    // Copiar resultados al host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, K * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Top " << K << " elementos más grandes:\n";
    for (int i = 0; i < K; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
