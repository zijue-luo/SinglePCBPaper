import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

input_csv = "20260115_162052_DOE_results_table.csv"
output_csv = "20260115_162052_lifetime.csv"

# 你现在如果删了 3000us，通常是 8；没删就是 9
# 也可以让它自动推断：block_size = df["wait_time"].nunique()
# 但我建议你明确写死，避免有缺点时误判。
BLOCK_SIZE = 8   # 或 9

def model(t_ms, A, tau_ms, b):
    return A * np.exp(-t_ms / tau_ms) + b

df = pd.read_csv(input_csv)

# --- 1) 给每行分配 run_id：按“块”切分，不合并重复点 ---
run_id = 0
current_key = None
seen_wt = set()
run_ids = []

for _, r in df.iterrows():
    key = (float(r["Ex"]), float(r["Ey"]), float(r["Ez"]))
    wt = int(r["wait_time"])

    start_new = False
    if current_key is None:
        start_new = True
    elif key != current_key:
        start_new = True
    elif len(seen_wt) >= BLOCK_SIZE:
        # 即使 key 没变，也强制切 run（防止相同 E 的重复块挨在一起被误合并）
        start_new = True

    if start_new:
        run_id += 1
        current_key = key
        seen_wt = set()

    run_ids.append(run_id)
    seen_wt.add(wt)

df["run_id"] = run_ids

# --- 2) 每个 run 内按 wait_time 聚合（防止同一 wait_time 重复行） ---
g = df.groupby(["run_id", "Ex", "Ey", "Ez", "wait_time"], as_index=False).agg(
    trapped=("trapped_signal", "sum"),
    loading=("loading_signal", "sum"),
)

g["p"] = g["trapped"] / g["loading"]
g["t_ms"] = g["wait_time"] / 1000.0

# over-dispersion：你说总方差 ~ 泊松的 2~3 倍
f_over = 2.5

rows = []
for (rid, ex, ey, ez), sub in g.groupby(["run_id", "Ex", "Ey", "Ez"]):
    sub = sub.sort_values("t_ms")

    t = sub["t_ms"].to_numpy(float)
    p = sub["p"].to_numpy(float)
    n = sub["loading"].to_numpy(float)

    # 权重（近似）：Var(p) ~ f * p(1-p)/n
    sigma = np.sqrt(np.maximum(f_over * p * np.maximum(1 - p, 1e-9) / np.maximum(n, 1.0), 1e-12))

    b0 = float(np.clip(np.min(p), 0.0, 1.0))
    A0 = float(np.clip(np.max(p) - b0, 1e-6, 1.0))
    tau0 = 0.8  # ms 初值

    try:
        popt, pcov = curve_fit(
            model, t, p,
            p0=[A0, tau0, b0],
            sigma=sigma, absolute_sigma=True,
            bounds=([0.0, 0.01, 0.0], [1.0, 100.0, 1.0]),
            maxfev=20000
        )
        A_fit, tau_fit, b_fit = popt
        tau_se = float(np.sqrt(np.diag(pcov))[1])
        p0_fit = float(A_fit + b_fit)
        ok = 1
    except Exception:
        A_fit = tau_fit = b_fit = tau_se = p0_fit = np.nan
        ok = 0

    rows.append({
        "run_id": int(rid),
        "Ex": ex, "Ey": ey, "Ez": ez,
        "n_wait": int(len(sub)),
        "tau_ms": tau_fit,
        "tau_se": tau_se,
        "p0_fit": p0_fit,
        "b_fit": b_fit,
        "fit_ok": ok,
        "total_loading": float(sub["loading"].sum()),
        "total_trapped": float(sub["trapped"].sum()),
    })

out = pd.DataFrame(rows).sort_values("run_id")

# 给同一 (Ex,Ey,Ez) 的重复测量编号：rep = 1,2,3...
out["rep"] = out.groupby(["Ex","Ey","Ez"]).cumcount() + 1

out.to_csv(output_csv, index=False)
print("Wrote:", output_csv, "rows=", len(out))
