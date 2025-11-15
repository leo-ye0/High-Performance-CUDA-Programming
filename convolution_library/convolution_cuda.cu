#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void create_edge_filter(unsigned int *filter, int N) {
    if (N == 3) {
        unsigned int edge[9] = {4294967295, 4294967295, 4294967295, 4294967295, 8, 4294967295, 4294967295, 4294967295, 4294967295};
        for (int i = 0; i < 9; i++) {
            filter[i] = edge[i];
        }
    } else {
        for (int i = 0; i < N * N; i++) {
            filter[i] = 1;
        }
    }
}

void create_test_image(unsigned int *image, int M) {
    for (int i = 0; i < M * M; i++) {
        image[i] = rand() % 256;
    }
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
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

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <image_size> <filter_size>\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    
    unsigned int *image = (unsigned int*)malloc(M * M * sizeof(unsigned int));
    unsigned int *filter = (unsigned int*)malloc(N * N * sizeof(unsigned int));
    unsigned int *result = (unsigned int*)malloc(M * M * sizeof(unsigned int));
    
    srand(42);
    create_test_image(image, M);
    create_edge_filter(filter, N);
    
    // Include full execution time including GPU setup
    double start = getTime();
    gpu_convolution(image, filter, result, M, N);
    double end = getTime();
    
    printf("%d,%d,%f\n", M, N, end - start);
    
    free(image);
    free(filter);
    free(result);
    
    return 0;
}