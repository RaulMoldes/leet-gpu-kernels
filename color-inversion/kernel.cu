#include <cuda_runtime.h>


__global__ void invert_kernel(unsigned char* image, int width, int height) {

    // Tile de 16x16 píxeles en memoria compartida
    __shared__ unsigned char tile[16][16][4];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Cargar datos a memoria compartida
    if (x < width && y < height) {
        int global_idx = (y * width + x) * 4;
        tile[ty][tx][0] = image[global_idx + 0];
        tile[ty][tx][1] = image[global_idx + 1];
        tile[ty][tx][2] = image[global_idx + 2];
        tile[ty][tx][3] = image[global_idx + 3];
    }

    __syncthreads();

    // Procesar en memoria compartida
    if (x < width && y < height) {
        tile[ty][tx][0] = 255 - tile[ty][tx][0]; // R
        tile[ty][tx][1] = 255 - tile[ty][tx][1]; // G
        tile[ty][tx][2] = 255 - tile[ty][tx][2]; // B
        // tile[ty][tx][3] permanece igual (alfa)
    }

    __syncthreads();

    // Escribir de vuelta
    if (x < width && y < height) {
        int global_idx = (y * width + x) * 4;
        image[global_idx + 0] = tile[ty][tx][0];
        image[global_idx + 1] = tile[ty][tx][1];
        image[global_idx + 2] = tile[ty][tx][2];
        image[global_idx + 3] = tile[ty][tx][3];
    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);
    invert_kernel << <gridSize, blockSize >> > (image, width, height);

    cudaDeviceSynchronize();
}
