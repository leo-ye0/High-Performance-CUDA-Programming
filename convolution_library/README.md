# CUDA Convolution Library

## Overview
Implements 2D convolution for image processing with CPU and CUDA versions, plus Python integration.

## Files
- `convolution_cpu.c` - CPU implementation
- `convolution_cuda.cu` - CUDA implementation with main function
- `convolution_lib.cu` - CUDA library for Python
- `libconvolution.so` - Compiled shared library
- `python_convolution.py` - Python interface
- `benchmark_convolution.sh` - Performance comparison script

## Features
- Edge detection filter (3x3 kernel)
- Supports various image and filter sizes
- Python ctypes integration
- Performance benchmarking

## Usage
```bash
# Benchmark CPU vs CUDA
bash benchmark_convolution.sh

# Test Python interface
python3 python_convolution.py
```

## Performance Results
CUDA shows mixed results due to overhead for small problems:
- Small images: CPU faster due to GPU setup overhead
- Large images: CUDA becomes competitive
- Python interface adds additional overhead but maintains GPU acceleration

## Key Findings
- GPU overhead significant for small convolutions
- CUDA benefits increase with larger image sizes
- Python integration successful with ctypes interface