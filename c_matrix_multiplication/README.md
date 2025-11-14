# Matrix multiplication benchmark (CPU)

Files added:

- `matrix_cpu.c` - C program that multiplies two N x N matrices on CPU. Usage: `./matrix_cpu N [trials]` and prints `N,avg_time_seconds`.
- `benchmark.sh` - Shell script to compile and run `matrix_cpu` for several N values and save results to `results.csv`.
- `plot_results.py` - Python script to plot `results.csv` and saves `results.png`.
- `requirements.txt` - Python requirements (matplotlib).

Quick start:

1. Make sure you have `gcc` and `python3` installed.
2. (Optional) Create a Python venv and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the benchmark (this will compile and run multiple sizes):

```bash
chmod +x benchmark.sh
./benchmark.sh
```

4. Plot results:

```bash
python3 plot_results.py results.csv
open results.png   # macOS-specific
```

Notes:
- The program uses a straightforward O(N^3) triple loop. For larger `N` this will become slow; adjust sizes in `benchmark.sh` as needed.
- You can pass number of trials to `matrix_cpu` as a second argument to adjust averaging.
