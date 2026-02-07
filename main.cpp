#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <numeric>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

static inline double now_sec() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// -------------------- SUM (reduction) --------------------
double sum_sequential(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x;
    return s;
}

double sum_threaded(const std::vector<double>& a, int nt) {
    if (nt < 1) nt = 1;
    std::vector<std::thread> th;
    std::vector<double> partial(nt, 0.0);

    size_t n = a.size();
    size_t chunk = n / (size_t)nt;

    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            size_t start = (size_t)t * chunk;
            size_t end = (t == nt - 1) ? n : start + chunk;
            double local = 0.0;
            for (size_t i = start; i < end; ++i) local += a[i];
            partial[t] = local;
        });
    }
    for (auto& x : th) x.join();

    return std::accumulate(partial.begin(), partial.end(), 0.0);
}

double sum_openmp(const std::vector<double>& a, int nt) {
#ifdef _OPENMP
    omp_set_num_threads(nt);
    double s = 0.0;
    #pragma omp parallel for reduction(+:s)
    for (size_t i = 0; i < a.size(); ++i) s += a[i];
    return s;
#else
    return sum_sequential(a);
#endif
}

// -------------------- TRANSFORM (element-wise) --------------------
// Transform: b[i] = sin(a[i])  (дараа нь checksum хийж optimization-оос хамгаална)
double transform_sequential(const std::vector<double>& a, std::vector<double>& b) {
    double checksum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        b[i] = std::sin(a[i]);
        checksum += b[i];
    }
    return checksum;
}

double transform_threaded(const std::vector<double>& a, std::vector<double>& b, int nt) {
    if (nt < 1) nt = 1;

    std::vector<std::thread> th;
    std::vector<double> partial(nt, 0.0);

    size_t n = a.size();
    size_t chunk = n / (size_t)nt;

    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            size_t start = (size_t)t * chunk;
            size_t end = (t == nt - 1) ? n : start + chunk;
            double local = 0.0;
            for (size_t i = start; i < end; ++i) {
                b[i] = std::sin(a[i]);
                local += b[i];
            }
            partial[t] = local;
        });
    }
    for (auto& x : th) x.join();

    return std::accumulate(partial.begin(), partial.end(), 0.0);
}

double transform_openmp(const std::vector<double>& a, std::vector<double>& b, int nt) {
#ifdef _OPENMP
    omp_set_num_threads(nt);
    double checksum = 0.0;

    #pragma omp parallel for reduction(+:checksum)
    for (size_t i = 0; i < a.size(); ++i) {
        b[i] = std::sin(a[i]);
        checksum += b[i];
    }
    return checksum;
#else
    return transform_sequential(a, b);
#endif
}

// -------------------- timing helper (mean of repeats) --------------------
template <class F>
double mean_time(int repeats, F&& f) {
    double total = 0.0;
    for (int i = 0; i < repeats; ++i) {
        auto t0 = Clock::now();
        f();
        auto t1 = Clock::now();
        total += std::chrono::duration<double>(t1 - t0).count();
    }
    return total / repeats;
}

int main() {
    const size_t N = 80'000'000;
    const int max_threads = 15;
    const int warmup_runs = 1;
    const int repeats = 3;

    std::vector<double> A(N, 1.0);
    std::vector<double> B(N, 0.0);

    // Warm-up (CPU, cache, runtime setup)
    for (int i = 0; i < warmup_runs; ++i) {
        (void)sum_sequential(A);
        (void)transform_sequential(A, B);
    }

    std::ofstream out("results.csv");
    out << "task,method,threads,time_sec\n";

    // ---- measure for each thread count ----
    for (int nt = 1; nt <= max_threads; ++nt) {
        // SUM
        double t_sum_thread = mean_time(repeats, [&]{ (void)sum_threaded(A, nt); });
        double t_sum_omp    = mean_time(repeats, [&]{ (void)sum_openmp(A, nt); });

        out << "sum,std::thread," << nt << "," << t_sum_thread << "\n";
        out << "sum,OpenMP,"     << nt << "," << t_sum_omp    << "\n";

        // TRANSFORM
        double t_tr_thread = mean_time(repeats, [&]{ (void)transform_threaded(A, B, nt); });
        double t_tr_omp    = mean_time(repeats, [&]{ (void)transform_openmp(A, B, nt); });

        out << "transform,std::thread," << nt << "," << t_tr_thread << "\n";
        out << "transform,OpenMP,"     << nt << "," << t_tr_omp    << "\n";

        std::cout << "Done nt=" << nt << "\n";
    }

    std::cout << "Saved results.csv\n";
    return 0;
}
