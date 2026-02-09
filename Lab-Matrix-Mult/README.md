# Lab – Matrix Multiplication (Sequential / std::thread / OpenMP)

## Зорилго
NxN хэмжээст float матрицуудыг үржүүлж:
- Sequential
- std::thread (олон thread)
- OpenMP

аргуудаар гүйцэтгэж **time, speedup, efficiency**-г харьцуулах.

## Файлууд
- `matmul_ac.cpp` – AC (plugged-in) дээр ажиллуулах хувилбар
- `matmul_battery.cpp` – Battery дээр ажиллуулах хувилбар
- `plot_15threads_auto.py` – 1–15 thread-ийн үр дүнг графикаар гаргана (CPU/cache info-г `lscpu`-с автоматаар уншина)

## Compile
```bash
g++ -O2 matmul_ac.cpp -o ac -fopenmp -pthread
g++ -O2 matmul_battery.cpp -o battery -fopenmp -pthread
