import pandas as pd
import numpy as np
import os

# ==================== 配置路径 ====================
WIN7_BASE = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\branches_misses_rate\win7"
WIN10_BASE = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\branches_misses_rate\win10"

# 阶段划分
PHASE_START = list(range(1, 21))  # 0-2s
PHASE_KEY = list(range(21, 51))  # 2-5s
PHASE_ENCRYPT = list(range(51, 100))  # 5-10s

PHASES = {
    "startup": PHASE_START,
    "key_gen": PHASE_KEY,
    "encrypt": PHASE_ENCRYPT
}


# ==================== 计算分支误预测率 ====================
def compute_branch_misprediction_rate(branches_file, misses_file):
    """计算分支误预测率"""
    df_branch = pd.read_csv(branches_file, encoding='latin-1')
    df_miss = pd.read_csv(misses_file, encoding='latin-1')

    # 获取标签
    labels = df_branch["label"].values

    results = []

    for phase_name, phase_cols in PHASES.items():
        # 累加阶段内所有采样点
        total_branches = df_branch[[f"data_{i}" for i in phase_cols]].sum(axis=1).values.astype(float)
        total_misses = df_miss[[f"data_{i}" for i in phase_cols]].sum(axis=1).values.astype(float)

        # 计算每个样本的误预测率（避免除零）
        rate = np.zeros_like(total_misses, dtype=float)
        mask = total_branches != 0
        rate[mask] = total_misses[mask] / total_branches[mask]

        # 按标签分组计算平均值（转换为百分比）
        mal_mask = (labels == 1)
        ben_mask = (labels == 0)

        mal_rate = np.mean(rate[mal_mask]) * 100
        ben_rate = np.mean(rate[ben_mask]) * 100

        results.append({
            "phase": phase_name,
            "malicious_rate": round(mal_rate, 2),
            "benign_rate": round(ben_rate, 2),
            "diff": round(mal_rate - ben_rate, 2)
        })

    return pd.DataFrame(results)


# ==================== 保存 CSV ====================
def save_results(df, filename):
    if df is not None:
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存: {filename}")


# ==================== Win7 计算 ====================
print("=" * 50)
print("Win7 分支误预测率")
print("=" * 50)

win7_branches = os.path.join(WIN7_BASE, "branches.csv")
win7_misses = os.path.join(WIN7_BASE, "branch-misses.csv")

if os.path.exists(win7_branches) and os.path.exists(win7_misses):
    win7_results = compute_branch_misprediction_rate(win7_branches, win7_misses)
    print(win7_results.to_string(index=False))
    save_results(win7_results, "win7_branch_misprediction.csv")
else:
    print("Win7 文件不存在，跳过")
    win7_results = None

# ==================== Win10 计算 ====================
print("\n" + "=" * 50)
print("Win10 分支误预测率 (P-core)")
print("=" * 50)

win10_core_branches = os.path.join(WIN10_BASE, "cpu_core_branches_.csv")
win10_core_misses = os.path.join(WIN10_BASE, "cpu_core_branch-misses_.csv")

if os.path.exists(win10_core_branches) and os.path.exists(win10_core_misses):
    win10_core_results = compute_branch_misprediction_rate(win10_core_branches, win10_core_misses)
    print("P-core:")
    print(win10_core_results.to_string(index=False))
    save_results(win10_core_results, "win10_pcore_branch_misprediction.csv")
else:
    print("Win10 P-core 文件不存在，跳过")
    win10_core_results = None

print("\n" + "=" * 50)
print("Win10 分支误预测率 (E-core)")
print("=" * 50)

win10_atom_branches = os.path.join(WIN10_BASE, "cpu_atom_branches_.csv")
win10_atom_misses = os.path.join(WIN10_BASE, "cpu_atom_branch-misses_.csv")

if os.path.exists(win10_atom_branches) and os.path.exists(win10_atom_misses):
    win10_atom_results = compute_branch_misprediction_rate(win10_atom_branches, win10_atom_misses)
    print("E-core:")
    print(win10_atom_results.to_string(index=False))
    save_results(win10_atom_results, "win10_ecore_branch_misprediction.csv")
else:
    print("Win10 E-core 文件不存在，跳过")
    win10_atom_results = None

print("\n✅ 所有结果已保存为 CSV 文件")