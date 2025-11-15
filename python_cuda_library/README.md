# Python-CUDA Library Integration

## Overview
Demonstrates how to compile CUDA kernels into a shared library and call them from Python using ctypes.

## Files
- `matrix_lib.cu` - CUDA library with C interface
- `libmatrix.so` - Compiled shared library
- `python_test.py` - Python script using ctypes

## Compilation
```bash
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so
```

## Usage
```bash
python3 python_test.py
```

## Key Features
- `extern "C"` interface for Python compatibility
- ctypes integration with numpy arrays
- Memory management handled in C/CUDA layer
- Tests 1024x1024 matrix multiplication