import ctypes
import numpy as np
import time

# Load shared library
lib = ctypes.cdll.LoadLibrary("./libconvolution.so")

# Define argument types
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

def create_edge_filter(N):
    if N == 3:
        return np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=np.uint32)
    return np.ones(N*N, dtype=np.uint32)

def test_convolution(M, N):
    # Create test image and filter
    image = np.random.randint(0, 256, M*M, dtype=np.uint32)
    filter_kernel = create_edge_filter(N)
    result = np.zeros(M*M, dtype=np.uint32)
    
    start = time.time()
    lib.gpu_convolution(image, filter_kernel, result, M, N)
    end = time.time()
    
    print(f"M={M}, N={N}: Python-CUDA convolution completed in {end - start:.6f} seconds")
    return end - start

if __name__ == "__main__":
    print("Testing CUDA Convolution from Python")
    print("=" * 50)
    
    # Test different sizes
    sizes = [(256, 3), (512, 3), (1024, 3), (256, 5), (512, 5)]
    for M, N in sizes:
        test_convolution(M, N)