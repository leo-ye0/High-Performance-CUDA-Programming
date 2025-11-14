#!/bin/bash
nvcc -arch=sm_75 matrix_gpu.cu -o matrix_gpu -O2
echo "N,time" > results.csv
./matrix_gpu 128 >> results.csv
./matrix_gpu 256 >> results.csv
./matrix_gpu 512 >> results.csv
./matrix_gpu 1024 >> results.csv
./matrix_gpu 1536 >> results.csv
./matrix_gpu 2048 >> results.csv
./matrix_gpu 3072 >> results.csv
./matrix_gpu 4096 >> results.csv
python3 plot_results.py