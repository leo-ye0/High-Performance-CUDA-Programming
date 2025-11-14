# Performance Comparison Analysis

## Complete Results (All Matrix Sizes)

| Implementation | N=128 | N=256 | N=512 | N=1024 | N=1536 | N=2048 | N=3072 | N=4096 |
|---------------|-------|-------|-------|--------|--------|--------|--------|--------|
| CPU (C) | 0.003s | 0.022s | 0.114s | 0.943s | 3.166s | 17.457s | 71.476s | 204.404s |
| Naive CUDA | 0.021ms | 0.081ms | 0.507ms | 3.670ms | 12.779ms | 33.682ms | 120.929ms | 284.910ms |
| Optimized CUDA | 0.026ms | 0.055ms | 0.292ms | 2.218ms | 7.803ms | 18.788ms | 79.760ms | 184.784ms |

## Speedup Analysis

**GPU vs CPU Speedup:**
- Naive CUDA: 133x - 717x speedup
- Optimized CUDA: 107x - 1106x speedup

**Optimization Impact:**
- Tiled implementation: 0.8x - 1.8x improvement over naive
- Best improvement at N=2048: 1.8x faster
- Consistent 1.5-1.7x improvement for large matrices

## Key Findings
- Shared memory tiling provides significant benefits for larger matrices
- Optimized CUDA achieves over 1000x speedup at N=4096
- Performance visualization available in `performance_comparison.png`