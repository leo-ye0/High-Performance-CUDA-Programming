# Optimized CUDA Matrix Multiplication - Shared Memory Tiling

## Overview
Implements shared memory tiling to reduce global memory accesses and improve performance.

## Key Optimizations
- 16x16 tile size for optimal shared memory usage
- Coalesced memory access patterns
- Reduced global memory bandwidth requirements

## Files
- `matrix_gpu_tiled.cu` - Tiled CUDA implementation
- `benchmark.sh` - Performance testing script

## Usage
```bash
bash benchmark.sh
```

## Performance Results
Benchmark generates `results.csv` with timing data for matrix sizes 128-4096 and `results.png` plot.