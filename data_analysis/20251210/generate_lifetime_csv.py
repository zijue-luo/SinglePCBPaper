import csv
import random

# 1. 定义所有不同的 wait_time（单位 us）
wait_times = []

# 0–2000 us：步进 100 us
wait_times += list(range(0, 2001, 100))       # 0, 100, ..., 2000

# 2400–5000 us：步进 400 us
wait_times += [2400, 2800, 3200, 3600, 4000, 4400, 4800, 5000]

# 长尾区
wait_times += [7000, 12000, 20000, 35000, 50000]

# 去重 + 排序（理论上本来就不重，这里保险）
wait_times = sorted(set(wait_times))
print("Unique wait_time points:", len(wait_times))   # 应该是 34

# 2. 每个点 repeat = 9
rows = []
repeat_per_point = 9
for w in wait_times:
    for _ in range(repeat_per_point):
        rows.append(w)
print("Total experiments (rows):", len(rows))        # 应该是 340

# 3. 打乱顺序（shuffle），方便后续 detrend
random.seed(42)   # 想每次都变就注释掉这一行
random.shuffle(rows)

# 4. 写出 CSV
filename = "lifetime_wait_time_design.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    # 表头：1 个 setpoint 列 + 一堆 data 列
    writer.writerow([
        "wait_time",        # 本次唯一 setpoint（单位 us）
        "act_RF_amplitude", # data 列（开始为 NaN / 空，跑完实验由脚本写）
        "ratio_signal",
        "ratio_lost",
        "loading_signal",
        "trapped_signal",
        "lost_signal",
        "time_cost",
    ])

    # 每行：先填 wait_time，其余 data 列留空
    for w in rows:
        writer.writerow([w, "", "", "", "", "", "", ""])

print(f"Saved to {filename}")
