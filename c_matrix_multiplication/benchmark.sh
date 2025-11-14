#!/bin/bash
gcc matrix_cpu.c -o matrix_cpu -O2
echo "N,time" > results.csv
./matrix_cpu 128 >> results.csv
./matrix_cpu 256 >> results.csv
./matrix_cpu 512 >> results.csv
./matrix_cpu 1024 >> results.csv
./matrix_cpu 1536 >> results.csv
./matrix_cpu 2048 >> results.csv
./matrix_cpu 3072 >> results.csv
./matrix_cpu 4096 >> results.csv
python3 plot_results.py