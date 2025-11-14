#!/bin/bash
nvcc -arch=sm_75 matrix_gpu_tiled.cu -o matrix_gpu_tiled -O2
echo "N,time" > results.csv
./matrix_gpu_tiled 128 >> results.csv
./matrix_gpu_tiled 256 >> results.csv
./matrix_gpu_tiled 512 >> results.csv
./matrix_gpu_tiled 1024 >> results.csv
./matrix_gpu_tiled 1536 >> results.csv
./matrix_gpu_tiled 2048 >> results.csv
./matrix_gpu_tiled 3072 >> results.csv
./matrix_gpu_tiled 4096 >> results.csv
python3 plot_results.py