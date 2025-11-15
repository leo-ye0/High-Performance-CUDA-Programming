#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolution_kernel(unsigned int *image, unsigned int *filter, 
                                   unsigned int *result, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < M && j < M) {
        int pad = N / 2;
        int sum = 0;
        
        for (int fi = 0; fi < N; fi++) {
            for (int fj = 0; fj < N; fj++) {
                int ii = i - pad + fi;
                int jj = j - pad + fj;
                if (ii >= 0 && ii < M && jj >= 0 && jj < M) {
                    sum += image[ii * M + jj] * filter[fi * N + fj];
                }
            }
        }
        result[i * M + j] = sum > 0 ? sum : 0;
    }
}

extern "C" void gpu_convolution(unsigned int *h_image, unsigned int *h_filter, 
                                unsigned int *h_result, int M, int N) {
    size_t image_size = M * M * sizeof(unsigned int);
    size_t filter_size = N * N * sizeof(unsigned int);
    
    unsigned int *d_image, *d_filter, *d_result;
    cudaMalloc(&d_image, image_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_result, image_size);
    
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    convolution_kernel<<<gridSize, blockSize>>>(d_image, d_filter, d_result, M, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, image_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_result);
}