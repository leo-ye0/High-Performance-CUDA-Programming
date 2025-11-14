import csv
import matplotlib.pyplot as plt

# Read results from all implementations
cpu_times = {}
with open('../c_matrix_multiplication/results.csv') as f:
    next(csv.reader(f))
    for row in csv.reader(f):
        cpu_times[int(row[0])] = float(row[1])

naive_times = {}
with open('../cuda_matrix_multiplication/results.csv') as f:
    next(csv.reader(f))
    for row in csv.reader(f):
        naive_times[int(row[0])] = float(row[1])

optimized_times = {}
with open('../optimized_cuda_matrix_multiplication/results.csv') as f:
    next(csv.reader(f))
    for row in csv.reader(f):
        optimized_times[int(row[0])] = float(row[1])

# All available sizes
sizes = sorted(set(cpu_times.keys()) & set(naive_times.keys()) & set(optimized_times.keys()))

print("Performance Comparison Table")
print("=" * 100)
print(f"{'Implementation':<15} ", end="")
for size in sizes:
    print(f"{'N='+str(size):<12}", end="")
print()
print("-" * 100)

# CPU times (seconds)
print(f"{'CPU (C)':<15} ", end="")
for size in sizes:
    print(f"{cpu_times[size]:<12.3f}", end="")
print()

# Naive CUDA times (ms)
print(f"{'Naive CUDA':<15} ", end="")
for size in sizes:
    print(f"{naive_times[size]*1000:<12.3f}", end="")
print()

# Optimized CUDA times (ms)
print(f"{'Optimized CUDA':<15} ", end="")
for size in sizes:
    print(f"{optimized_times[size]*1000:<12.3f}", end="")
print()

print("\nSpeedup Analysis")
print("=" * 60)
for size in sizes:
    naive_speedup = cpu_times[size] / naive_times[size]
    opt_speedup = cpu_times[size] / optimized_times[size]
    improvement = naive_times[size] / optimized_times[size]
    print(f"N={size}: Naive={naive_speedup:.0f}x, Optimized={opt_speedup:.0f}x, Improvement={improvement:.1f}x")

# Create comparison plot
N = list(sizes)
cpu_vals = [cpu_times[n] for n in N]
naive_vals = [naive_times[n] for n in N]
opt_vals = [optimized_times[n] for n in N]

plt.figure(figsize=(10, 6))
plt.loglog(N, cpu_vals, 'o-', label='CPU (C)')
plt.loglog(N, naive_vals, 's-', label='Naive CUDA')
plt.loglog(N, opt_vals, '^-', label='Optimized CUDA')
plt.xlabel('Matrix Size N')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: CPU vs CUDA Implementations')
plt.legend()
plt.grid(True)
plt.savefig('performance_comparison.png')
print('\nComparison plot saved to performance_comparison.png')

# Save table to file
with open('performance_table.txt', 'w') as f:
    f.write("Performance Comparison Table\n")
    f.write("=" * 100 + "\n")
    f.write(f"{'Implementation':<15} ")
    for size in sizes:
        f.write(f"{'N='+str(size):<12}")
    f.write("\n")
    f.write("-" * 100 + "\n")
    
    # CPU times (seconds)
    f.write(f"{'CPU (C)':<15} ")
    for size in sizes:
        f.write(f"{cpu_times[size]:<12.3f}")
    f.write("\n")
    
    # Naive CUDA times (ms)
    f.write(f"{'Naive CUDA':<15} ")
    for size in sizes:
        f.write(f"{naive_times[size]*1000:<12.3f}")
    f.write("\n")
    
    # Optimized CUDA times (ms)
    f.write(f"{'Optimized CUDA':<15} ")
    for size in sizes:
        f.write(f"{optimized_times[size]*1000:<12.3f}")
    f.write("\n\n")
    
    f.write("Speedup Analysis\n")
    f.write("=" * 60 + "\n")
    for size in sizes:
        naive_speedup = cpu_times[size] / naive_times[size]
        opt_speedup = cpu_times[size] / optimized_times[size]
        improvement = naive_times[size] / optimized_times[size]
        f.write(f"N={size}: Naive={naive_speedup:.0f}x, Optimized={opt_speedup:.0f}x, Improvement={improvement:.1f}x\n")

print('Performance table saved to performance_table.txt')