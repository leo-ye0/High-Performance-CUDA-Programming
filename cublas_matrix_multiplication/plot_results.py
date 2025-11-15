import csv
import matplotlib.pyplot as plt

N, times = [], []
with open('results.csv') as f:
    next(csv.reader(f))  # skip header
    for row in csv.reader(f):
        N.append(int(row[0]))
        times.append(float(row[1]))

plt.plot(N, times, 'o-')
plt.xlabel('Matrix Size N')
plt.ylabel('Time (seconds)')
plt.title('cuBLAS Matrix Multiplication Performance')
plt.savefig('results.png')
print('Plot saved to results.png')