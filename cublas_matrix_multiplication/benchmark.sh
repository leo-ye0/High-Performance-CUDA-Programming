#!/bin/bash
nvcc -arch=sm_75 matrix_cublas.cu -o matrix_cublas -lcublas -O2
echo "N,time" > results.csv
./matrix_cublas 128 >> results.csv
./matrix_cublas 256 >> results.csv
./matrix_cublas 512 >> results.csv
./matrix_cublas 1024 >> results.csv
./matrix_cublas 1536 >> results.csv
./matrix_cublas 2048 >> results.csv
./matrix_cublas 3072 >> results.csv
./matrix_cublas 4096 >> results.csv
python3 plot_results.py