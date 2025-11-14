# CUDA Matrix Multiplication - Naïve Implementation

## Overview
This implementation demonstrates a basic CUDA kernel for matrix multiplication where each thread computes one element of the output matrix.

## Files
- `matrix_gpu.cu` - Main CUDA program with naïve kernel
- `benchmark.sh` - Automated benchmarking script
- `plot_results.py` - Performance visualization
- `requirements.txt` - Python dependencies

## Quick Start
```bash
# Run benchmark
bash benchmark.sh

# Or compile and run manually
nvcc -arch=sm_75 matrix_gpu.cu -o matrix_gpu
./matrix_gpu 1024
```

## Performance Results
Benchmark generates `results.csv` with timing data for matrix sizes 128-2048.

## Kernel Details
- Block size: 16x16 threads
- Each thread computes one output element
- Uses row-major memory layout