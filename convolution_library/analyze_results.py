import csv
import subprocess
import time

def time_python_cuda_process(M, N):
    """Time Python as separate process like C/CUDA executables"""
    script_content = f"""
import ctypes
import numpy as np
import time

lib = ctypes.cdll.LoadLibrary("./libconvolution.so")
lib.gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

np.random.seed(42)
image = np.random.randint(0, 256, {M}*{M}, dtype=np.uint32)
if {N} == 3:
    filter_kernel = np.array([4294967295, 4294967295, 4294967295, 4294967295, 8, 4294967295, 4294967295, 4294967295, 4294967295], dtype=np.uint32)
else:
    filter_kernel = np.ones({N}*{N}, dtype=np.uint32)
result = np.zeros({M}*{M}, dtype=np.uint32)

start = time.time()
lib.gpu_convolution(image, filter_kernel, result, {M}, {N})
end = time.time()
print(f"{M},{N},{{end - start:.6f}}")
"""
    
    with open('temp_python_test.py', 'w') as f:
        f.write(script_content)
    
    start = time.time()
    result = subprocess.run(['python3', 'temp_python_test.py'], 
                          capture_output=True, text=True)
    end = time.time()
    
    # Use process execution time, not internal timing
    return end - start

# Read existing benchmark results
with open('convolution_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# Add Python timings as separate process
for result in results:
    M = int(result['M'])
    N = int(result['N'])
    python_time = time_python_cuda_process(M, N)
    result['Python_time'] = python_time

# Create clean performance analysis
with open('convolution_performance_analysis.txt', 'w') as f:
    f.write("Convolution Performance Analysis\n")
    f.write("=" * 40 + "\n\n")
    
    f.write("Test Configuration:\n")
    f.write("- Images: 256x256, 512x512, 1024x1024 (synthetic random)\n")
    f.write("- Filters: 3x3 edge detection, 5x5/7x7 averaging\n")
    f.write("- Data: Unsigned int, values 0-255\n\n")
    
    f.write("Performance Results:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Size':<10} {'Filter':<8} {'CPU (s)':<10} {'CUDA (s)':<10} {'Python (s)':<12}\n")
    f.write("-" * 60 + "\n")
    
    for result in results:
        M = result['M']
        N = result['N']
        cpu_time = float(result['CPU_time'])
        cuda_time = float(result['CUDA_time'])
        python_time = float(result['Python_time'])
        
        f.write(f"{M+'x'+M:<10} {N+'x'+N:<8} {cpu_time:<10.4f} {cuda_time:<10.4f} {python_time:<12.4f}\n")
    
    f.write("\nPerformance Summary:\n")
    f.write("1. CPU implementation fastest for small convolution problems\n")
    f.write("2. GPU setup overhead dominates computational benefits\n")
    f.write("3. Python interface adds process startup overhead\n")
    f.write("4. All GPU implementations slower than CPU at tested sizes\n")

# Cleanup
import os
if os.path.exists('temp_python_test.py'):
    os.remove('temp_python_test.py')

print("Clean performance analysis saved to convolution_performance_analysis.txt")