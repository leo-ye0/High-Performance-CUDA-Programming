#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initializeMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    srand(42);
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Warm up
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();
    
    double startTime = getTime();
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();
    double endTime = getTime();
    
    double executionTime = endTime - startTime;
    printf("%d,%f\n", N, executionTime);
    
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}