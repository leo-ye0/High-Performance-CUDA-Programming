#!/bin/bash

# Compile programs
gcc convolution_cpu.c -o convolution_cpu -O2
nvcc convolution_cuda.cu -o convolution_cuda -O2

# Test configurations: M (image size), N (filter size)
echo "M,N,CPU_time,CUDA_time" > convolution_results.csv

# Test different image and filter sizes
for M in 256 512 1024; do
    for N in 3 5 7; do
        echo "Testing M=$M, N=$N"
        
        # CPU timing
        cpu_result=$(./convolution_cpu $M $N)
        cpu_time=$(echo $cpu_result | cut -d',' -f3)
        
        # CUDA timing  
        cuda_result=$(./convolution_cuda $M $N)
        cuda_time=$(echo $cuda_result | cut -d',' -f3)
        
        echo "$M,$N,$cpu_time,$cuda_time" >> convolution_results.csv
    done
done

echo "Benchmark complete. Results saved to convolution_results.csv"