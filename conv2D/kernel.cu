#include <cuda_runtime.h>


__device__ inline int get_global_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_global_idy() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}



extern "C" __global__ void conv2d_forward(
    const float* input,
    const float* filter,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    const int TILE_SIZE = 16;

    // Use shared memory for input and filter
    // Shared memory size: input tile + filter
    // Total shared memory size: (TILE_SIZE + kernel_height - 1) *
    extern __shared__ float shared_mem[];

    // Compute the size of the input tile
    // Input tile size is the size of the output tile plus the kernel size minus 1
    // This accounts for the padding needed for the convolution
    int input_tile_h = TILE_SIZE + kernel_height - 1;
    int input_tile_w = TILE_SIZE + kernel_width - 1;
    int filter_size = kernel_height * kernel_width;

    // Partition shared memory
    // First part for input tile, second part for filter
    // shared_mem size: input_tile_h * input_tile_w + filter_size
    float* shared_input = shared_mem;
    float* shared_filter = shared_mem + input_tile_h * input_tile_w;

    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_c = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    // Compute the output coordinates
    bool valid_output = (out_x < out_width && out_y < out_height &&
        out_c < out_channels && batch_idx < batch_size);

    float result = 0.0f;

    // Process each input channel
    for (int in_c = 0; in_c < in_channels; in_c++) {

        // 1. Load the filter cooperatively into shared memory
        // Each thread loads one element of the filter
        for (int i = thread_id; i < filter_size; i += threads_per_block) {
            int ky = i / kernel_width;
            int kx = i % kernel_width;

            int filter_idx = out_c * (in_channels * kernel_height * kernel_width) +
                in_c * (kernel_height * kernel_width) +
                ky * kernel_width + kx;
            shared_filter[i] = filter[filter_idx];
        }

        // 2. Load the input tile into shared memory
        // Each thread loads one element of the input tile
        int input_elements = input_tile_h * input_tile_w;
        for (int i = thread_id; i < input_elements; i += threads_per_block) {
            int tile_y = i / input_tile_w;
            int tile_x = i % input_tile_w;

            // Calculate the input coordinates based on the tile position
            int in_x = blockIdx.x * TILE_SIZE * stride_w - pad_w + tile_x;
            int in_y = blockIdx.y * TILE_SIZE * stride_h - pad_h + tile_y;

            if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                int input_idx = batch_idx * (in_channels * in_height * in_width) +
                    in_c * (in_height * in_width) +
                    in_y * in_width + in_x;
                shared_input[i] = input[input_idx];
            }
            else {
                shared_input[i] = 0.0f; // Zero padding
            }
        }

        __syncthreads();

        // 3. Perform the convolution operation
        // Each thread computes a part of the output
        if (valid_output) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    // Position in the shared memory tile
                    int shared_y = ty * stride_h + ky;
                    int shared_x = tx * stride_w + kx;

                    // Verify if the shared memory indices are within bounds
                    if (shared_y < input_tile_h && shared_x < input_tile_w) {
                        int input_idx = shared_y * input_tile_w + shared_x;
                        int filter_idx = ky * kernel_width + kx;

                        result += shared_input[input_idx] * shared_filter[filter_idx];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Add bias if provided
    // Bias is added only if the output is valid
    if (valid_output) {
        if (bias != nullptr) {
            result += bias[out_c];
        }

        int output_idx = batch_idx * (out_channels * out_height * out_width) +
            out_c * (out_height * out_width) +
            out_y * out_width + out_x;
        output[output_idx] = result;
    }

}


extern "C" void solve(const float* input, const float* kernel, float* output,
    int input_rows, int input_cols,
    int kernel_rows, int kernel_cols) {
    // Parameters
    int batch_size = 1;
    int in_channels = 1;
    int out_channels = 1;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;

    int in_height = input_rows;
    int in_width = input_cols;
    int out_height = (in_height + 2 * pad_h - kernel_rows) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_cols) / stride_w + 1;

    // Sizes
    size_t input_size = batch_size * in_channels * in_height * in_width * sizeof(float);
    size_t filter_size = out_channels * in_channels * kernel_rows * kernel_cols * sizeof(float);
    size_t output_size = batch_size * out_channels * out_height * out_width * sizeof(float);
    size_t bias_size = out_channels * sizeof(float);

    // Allocate memory
    float* d_input, * d_filter, * d_output, * d_bias;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMemset(d_bias, 0, bias_size);  // Use zero bias for now

    // Copy input & kernel
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, kernel, filter_size, cudaMemcpyHostToDevice);

    // Kernel configuration
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels);

    int input_tile_h = TILE_SIZE + kernel_rows - 1;
    int input_tile_w = TILE_SIZE + kernel_cols - 1;
    int shared_mem_size = (input_tile_h * input_tile_w + kernel_rows * kernel_cols) * sizeof(float);

    // Launch kernel
    conv2d_forward << <gridDim, blockDim, shared_mem_size >> > (
        d_input, d_filter, d_bias, d_output,
        batch_size, in_channels,
        in_height, in_width,
        out_channels, out_height, out_width,
        kernel_rows, kernel_cols,
        stride_h, stride_w,
        pad_h, pad_w
        );

    // Copy result back
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_bias);
}
