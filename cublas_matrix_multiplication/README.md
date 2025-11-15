# cuBLAS Matrix Multiplication

## Overview
Uses NVIDIA's cuBLAS library for highly optimized matrix multiplication on GPU.

## Key Features
- Leverages cuBLAS SGEMM for single-precision matrix multiplication
- Highly optimized NVIDIA implementation
- Minimal code required for maximum performance

## Files
- `matrix_cublas.cu` - cuBLAS implementation
- `benchmark.sh` - Performance testing script
- `plot_results.py` - Performance visualization

## Usage
```bash
bash benchmark.sh
```

## Performance Results
cuBLAS achieves exceptional performance:
- N=2048: 2.910ms (5999x speedup over CPU)
- N=4096: 31.941ms (6399x speedup over CPU)

Significantly outperforms custom CUDA implementations.