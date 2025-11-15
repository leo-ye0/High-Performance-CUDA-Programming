# High-Performance CUDA Programming

## Overview
Comprehensive CUDA programming project demonstrating matrix multiplication and convolution implementations with performance analysis.

## Project Structure

### Part 1: C Matrix Multiplication Baseline
- `c_matrix_multiplication/` - CPU implementation with benchmarking and visualization

### Part 2: Naive CUDA Implementation  
- `cuda_matrix_multiplication/` - Basic GPU parallelization with performance measurement

### Part 4: Optimized CUDA with Shared Memory
- `optimized_cuda_matrix_multiplication/` - Tiled implementation using shared memory optimization

### Part 6: cuBLAS Library Integration
- `cublas_matrix_multiplication/` - Highly optimized NVIDIA library implementation

### Part 5: Performance Comparison
- `performance_comparison/` - Comprehensive analysis across all implementations with tables and plots

### Part 7: Python-CUDA Integration
- `python_cuda_library/` - Shared library creation and Python ctypes interface

### Part 8: Convolution Library
- `convolution_library/` - 2D convolution with CPU/CUDA implementations and Python integration

## Key Results

**Matrix Multiplication Performance (N=2048):**
- CPU: 17.457 seconds
- Naive CUDA: 33.682ms (518x speedup)
- Optimized CUDA: 18.788ms (929x speedup) 
- cuBLAS: 2.910ms (5999x speedup)

**Convolution Performance (1024x1024, 3x3 filter):**
- CPU: 0.0164 seconds
- Direct CUDA: 0.1737 seconds
- Python-CUDA: 0.4223 seconds

## Shared Libraries for Python

**Matrix Multiplication Library (`libmatrix.so`):**
- Function: `gpu_matrix_multiply(float *A, float *B, float *C, int N)`
- Features: Shared memory tiling, automatic GPU memory management
- Performance: ~929x speedup over CPU for large matrices

**Convolution Library (`libconvolution.so`):**
- Function: `gpu_convolution(uint *image, uint *filter, uint *result, int M, int N)`
- Features: 2D convolution, edge detection filters, boundary handling
- Performance: CPU faster for small images due to GPU overhead

## Key Findings
- **Matrix Multiplication**: GPU provides 500-6000x acceleration over CPU
- **Shared Memory**: Tiling improves performance by 1.5-1.8x over naive CUDA
- **cuBLAS**: Outperforms custom kernels by 3-6x
- **Python Interface**: Adds 2-3x overhead but maintains GPU benefits
- **Convolution**: CPU faster for small problems due to GPU setup overhead

## Usage
Each folder contains benchmark scripts and documentation. Run `bash benchmark.sh` in respective directories for performance testing.

## Technologies
- CUDA C/C++
- cuBLAS library
- Python ctypes integration
- Performance visualization with matplotlib