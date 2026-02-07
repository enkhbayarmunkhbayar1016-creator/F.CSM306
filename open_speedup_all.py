import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE, "build", "results.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(BASE, "results.csv")

print("Reading:", csv_path)
df = pd.read_csv(csv_path)

OUT_DIR = os.path.join(BASE, "zuragnuud.png")
os.makedirs(OUT_DIR, exist_ok=True)

def compute_speedup(task, method):
    sub = df[(df["task"] == task) & (df["method"] == method)].copy().sort_values("threads")
    t1 = float(sub[sub["threads"] == 1]["time_sec"].values[0])
    sub["speedup"] = t1 / sub["time_sec"]
    return sub["threads"].values, sub["speedup"].values

# 4 шугам: thread/openmp + sum/transform(sin)
t_ts, s_ts = compute_speedup("sum", "std::thread")
t_tsin, s_tsin = compute_speedup("transform", "std::thread")
t_os, s_os = compute_speedup("sum", "OpenMP")
t_osin, s_osin = compute_speedup("transform", "OpenMP")

plt.figure(figsize=(10, 6))
plt.plot(t_ts, s_ts, marker="o", linewidth=2, label="threads-sum")
plt.plot(t_tsin, s_tsin, marker="o", linewidth=2, label="threads-sin")
plt.plot(t_os, s_os, marker="o", linewidth=2, label="openmp-sum")
plt.plot(t_osin, s_osin, marker="o", linewidth=2, label="openmp-sin")

plt.title("Speedup Comparison (All)")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

out_png = os.path.join(OUT_DIR, "speedup_comparison_all.png")
plt.savefig(out_png, dpi=200)
print("Saved:", out_png)

# зураг нээж харах (цонх гаргана)
plt.show()
