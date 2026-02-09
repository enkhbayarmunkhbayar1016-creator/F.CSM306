import re
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

MAX_T = 15

def get_lscpu_info():
    keys = [
        "Model name",
        "CPU(s)",
        "Core(s) per socket",
        "Thread(s) per core",
        "L1d cache",
        "L1i cache",
        "L2 cache",
        "L3 cache",
    ]
    info = {k: "?" for k in keys}

    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            if k in info:
                info[k] = v
    except Exception:
        pass

    return info

def detect_power_now():
    # 1 = AC plugged-in, 0 = battery (зарим ноут дээр нэр өөр байж болно)
    candidates = [
        "/sys/class/power_supply/AC/online",
        "/sys/class/power_supply/ACAD/online",
        "/sys/class/power_supply/Mains/online",
    ]
    for p in candidates:
        try:
            v = Path(p).read_text().strip()
            if v == "1":
                return "Plugged-in"
            if v == "0":
                return "Battery"
        except Exception:
            pass
    return "Unknown"

def read_bench_csv(path: str):
    """
    Expected C++ output format:
    N=512
    t_seq=...
    threads,t_thread,t_openmp
    1,....
    2,....
    ...
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").strip().splitlines()

    N = None
    t_seq = None
    rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("N="):
            N = int(line.split("=", 1)[1])
            continue

        if line.startswith("t_seq="):
            t_seq = float(line.split("=", 1)[1])
            continue

        if line.lower().startswith("threads,"):
            continue

        m = re.match(r"^\s*(\d+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*$", line)
        if m:
            T = int(m.group(1))
            t_thr = float(m.group(2))
            t_omp = float(m.group(3))
            if 1 <= T <= MAX_T:
                rows.append((T, t_thr, t_omp))

    if N is None or t_seq is None:
        raise ValueError(f"{path} файлд N=... эсвэл t_seq=... мөр байхгүй байна.")

    rows.sort(key=lambda x: x[0])

    if len(rows) == 0:
        raise ValueError(f"{path} файлд thread мөрүүд уншигдсангүй. (1, t_thread, t_openmp) формат шалга)")

    if len(rows) < MAX_T:
        print(f"⚠️ Анхаар: {path} дотор 1..{MAX_T} бүх мөр байхгүй байна. Олдсон: {len(rows)}")

    return N, t_seq, rows

def speedup(tseq, tpar):
    return tseq / tpar

def efficiency(sp, T):
    return sp / T

def annotate(ax, lscpu, N, power_note):
    text = (
        f"CPU: {lscpu['Model name']}\n"
        f"CPU(s): {lscpu['CPU(s)']} | Cores/socket: {lscpu['Core(s) per socket']} | Threads/core: {lscpu['Thread(s) per core']}\n"
        f"L1d: {lscpu['L1d cache']} | L1i: {lscpu['L1i cache']}\n"
        f"L2: {lscpu['L2 cache']} | L3: {lscpu['L3 cache']}\n"
        f"N (matrix size): {N}\n"
        f"Power: {power_note}"
    )
    ax.text(
        0.02, -0.34, text,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )

def main():
    # ---- Read system info once
    lscpu = get_lscpu_info()

    # ---- Read benchmark data
    N_ac, t_seq_ac, rows_ac = read_bench_csv("bench_ac.csv")
    N_ba, t_seq_ba, rows_ba = read_bench_csv("bench_battery.csv")

    # assume same N
    N = N_ac

    # Extract lists
    T_ac = [r[0] for r in rows_ac]
    t_thr_ac = [r[1] for r in rows_ac]
    t_omp_ac = [r[2] for r in rows_ac]

    T_ba = [r[0] for r in rows_ba]
    t_thr_ba = [r[1] for r in rows_ba]
    t_omp_ba = [r[2] for r in rows_ba]

    # Compute speedups
    sp_thr_ac = [speedup(t_seq_ac, t) for t in t_thr_ac]
    sp_omp_ac = [speedup(t_seq_ac, t) for t in t_omp_ac]

    sp_thr_ba = [speedup(t_seq_ba, t) for t in t_thr_ba]
    sp_omp_ba = [speedup(t_seq_ba, t) for t in t_omp_ba]

    # Compute efficiencies
    eff_thr_ac = [efficiency(sp, T) for sp, T in zip(sp_thr_ac, T_ac)]
    eff_omp_ac = [efficiency(sp, T) for sp, T in zip(sp_omp_ac, T_ac)]

    eff_thr_ba = [efficiency(sp, T) for sp, T in zip(sp_thr_ba, T_ba)]
    eff_omp_ba = [efficiency(sp, T) for sp, T in zip(sp_omp_ba, T_ba)]

    # Power note shown on graph (we label the data, not current state)
    power_note = "Battery vs Plugged-in"

    # =========================
    # 1) SPEEDUP graph
    # =========================
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(T_ba, sp_thr_ba, marker="o", linestyle="--", label="threads (Battery)")
    ax.plot(T_ba, sp_omp_ba, marker="o", linestyle=":",  label="OpenMP (Battery)")

    ax.plot(T_ac, sp_thr_ac, marker="o", label="threads (Plugged-in)")
    ax.plot(T_ac, sp_omp_ac, marker="o", label="OpenMP (Plugged-in)")

    ax.plot([1, MAX_T], [1, MAX_T], linestyle="--", label="Ideal (S=T)")

    ax.set_xlabel("Threads (1–15)")
    ax.set_ylabel("Speedup (T1 / Tn)")
    ax.set_title("matmul: Speedup (Battery vs Plugged-in)")
    ax.grid(True)
    ax.legend()

    annotate(ax, lscpu, N, power_note)
    fig.subplots_adjust(bottom=0.35)
    fig.savefig("speedup_1_15.png", dpi=200)

    # =========================
    # 2) EFFICIENCY graph
    # =========================
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(T_ba, eff_thr_ba, marker="o", linestyle="--", label="threads (Battery)")
    ax.plot(T_ba, eff_omp_ba, marker="o", linestyle=":",  label="OpenMP (Battery)")

    ax.plot(T_ac, eff_thr_ac, marker="o", label="threads (Plugged-in)")
    ax.plot(T_ac, eff_omp_ac, marker="o", label="OpenMP (Plugged-in)")

    ax.plot([1, MAX_T], [1, 1], linestyle="--", label="Ideal (E=1)")

    ax.set_xlabel("Threads (1–15)")
    ax.set_ylabel("Efficiency")
    ax.set_title("matmul: Efficiency (Battery vs Plugged-in)")
    ax.grid(True)
    ax.legend()

    annotate(ax, lscpu, N, power_note)
    fig.subplots_adjust(bottom=0.35)
    fig.savefig("efficiency_1_15.png", dpi=200)

    plt.show()
    print("✅ Saved: speedup_1_15.png, efficiency_1_15.png")

if __name__ == "__main__":
    main()
