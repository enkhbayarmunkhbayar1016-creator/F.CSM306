import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

# CSV path (build/ байвал тэндээс, үгүй бол root-оос)
csv_path = os.path.join(BASE, "build", "results.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(BASE, "results.csv")

df = pd.read_csv(csv_path)

# Output folder
OUT_DIR = os.path.join(BASE, "zuragnuud.png")
os.makedirs(OUT_DIR, exist_ok=True)

def plot_task(task_name, max_threads=15):
    sub = df[df["task"] == task_name].copy()
    if sub.empty:
        raise RuntimeError(f"No rows found for task='{task_name}'")

    # 1..max_threads range
    sub = sub[(sub["threads"] >= 1) & (sub["threads"] <= max_threads)]

    # T1 per method
    t1_map = {}
    for method in sub["method"].unique():
        t1 = sub[(sub["method"] == method) & (sub["threads"] == 1)]["time_sec"].values
        if len(t1) == 0:
            raise RuntimeError(f"T1 not found for {task_name} / {method}")
        t1_map[method] = float(t1[0])

    # speedup, efficiency
    sub["T1"] = sub["method"].map(t1_map)
    sub["speedup"] = sub["T1"] / sub["time_sec"]
    sub["eff"] = sub["speedup"] / sub["threads"]
    sub["eff_pct"] = sub["eff"] * 100.0

    threads = sorted(sub["threads"].unique())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Speedup
    for method in ["std::thread", "OpenMP"]:
        m = sub[sub["method"] == method].sort_values("threads")
        if m.empty:
            continue
        ax1.plot(m["threads"], m["speedup"], marker="o", linewidth=2, label=method)

    ax1.plot(threads, threads, "--", linewidth=2, label="Ideal (y=x)")
    ax1.set_title(f"{task_name}: Speedup")
    ax1.set_ylabel("Speedup (T1 / Tn)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    # Efficiency (%)
    for method in ["std::thread", "OpenMP"]:
        m = sub[sub["method"] == method].sort_values("threads")
        if m.empty:
            continue
        ax2.plot(m["threads"], m["eff_pct"], marker="o", linewidth=2, label=method)

    ax2.plot(threads, [100.0]*len(threads), "--", linewidth=2, label="Ideal (100%)")
    ax2.set_title(f"{task_name}: Efficiency")
    ax2.set_xlabel(f"Threads (1-{max_threads})")
    ax2.set_ylabel("Efficiency (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"{task_name}_speedup_eff.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

# Run
plot_task("transform", max_threads=15)
# plot_task("sum", max_threads=15)  # хүсвэл нээ
