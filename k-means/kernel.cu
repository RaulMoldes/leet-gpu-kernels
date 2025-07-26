#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__device__ __inline__ float euclidean_distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy; // Usamos distancia al cuadrado para eficiencia
}

__global__ void assign_points_and_accumulate(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    const float* __restrict__ centroid_x,
    const float* __restrict__ centroid_y,
    int* __restrict__ labels,
    float* __restrict__ sum_x,
    float* __restrict__ sum_y,
    int* __restrict__ count,
    int sample_size,
    int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;

    float x = data_x[idx];
    float y = data_y[idx];
    float min_dist = 1e30f;  // FLT_MAX no está definido fácilmente, ponemos un valor alto
    int best_centroid = 0;

    for (int i = 0; i < k; ++i) {
        float dist = euclidean_distance(x, y, centroid_x[i], centroid_y[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }

    labels[idx] = best_centroid;

    atomicAdd(&sum_x[best_centroid], x);
    atomicAdd(&sum_y[best_centroid], y);
    atomicAdd(&count[best_centroid], 1);
}

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
    float* initial_centroid_x, float* initial_centroid_y,
    float* final_centroid_x, float* final_centroid_y,
    int sample_size, int k, int max_iterations) {

    // Punteros internos para suma y conteo
    float* d_sum_x;
    float* d_sum_y;
    int* d_count;

    cudaMalloc(&d_sum_x, k * sizeof(float));
    cudaMalloc(&d_sum_y, k * sizeof(float));
    cudaMalloc(&d_count, k * sizeof(int));

    // Copiar centroides iniciales a final_centroid_* (in-place update)
    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

    int threadsPerBlock = 256;
    int blocks = (sample_size + threadsPerBlock - 1) / threadsPerBlock;

    // Buffer temporal para centroides en host para comprobar convergencia
    float* h_centroid_x = new float[k];
    float* h_centroid_y = new float[k];

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Reset accumulators
        cudaMemset(d_sum_x, 0, k * sizeof(float));
        cudaMemset(d_sum_y, 0, k * sizeof(float));
        cudaMemset(d_count, 0, k * sizeof(int));

        // Lanzar kernel
        assign_points_and_accumulate << <blocks, threadsPerBlock >> > (
            data_x, data_y,
            final_centroid_x, final_centroid_y,
            labels,
            d_sum_x, d_sum_y, d_count,
            sample_size, k
            );
        cudaDeviceSynchronize();

        // Copiar sumas y conteos a host
        float h_sum_x[k], h_sum_y[k];
        int h_count[k];
        cudaMemcpy(h_sum_x, d_sum_x, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sum_y, d_sum_y, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);

        // Copiar centroides actuales a host para verificar cambio
        cudaMemcpy(h_centroid_x, final_centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_centroid_y, final_centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost);

        // Actualizar centroides y chequear convergencia
        bool converged = true;
        for (int i = 0; i < k; ++i) {
            if (h_count[i] > 0) {
                float new_x = h_sum_x[i] / h_count[i];
                float new_y = h_sum_y[i] / h_count[i];
                if (fabs(new_x - h_centroid_x[i]) > 1e-4f || fabs(new_y - h_centroid_y[i]) > 1e-4f) {
                    converged = false;
                }
                h_centroid_x[i] = new_x;
                h_centroid_y[i] = new_y;
            }
        }

        // Copiar centroides actualizados a device
        cudaMemcpy(final_centroid_x, h_centroid_x, k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(final_centroid_y, h_centroid_y, k * sizeof(float), cudaMemcpyHostToDevice);

        if (converged) {
            break;
        }
    }

    delete[] h_centroid_x;
    delete[] h_centroid_y;

    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
}
