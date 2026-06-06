import pandas as pd
import numpy as np
import os

# ======================== 路径配置 ========================
base_win7 = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win7"
base_win10 = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win10"
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== 阶段划分 ========================
PHASE_START = list(range(1, 21))  # 0-2s (20个采样点)
PHASE_KEY = list(range(21, 51))  # 2-5s (30个采样点)
PHASE_ENCRYPT = list(range(51, 100))  # 5-10s (49个采样点)

# ======================== 特征配置 ========================
WIN7_FEATURES = [
    ("node-stores.csv", "node-stores_mean", "mean"),
    ("cache-misses.csv", "cache-misses_std", "std"),
    ("cpu-cycles.csv", "cpu-cycles_med", "med"),
    ("node-stores.csv", "node-stores_std", "std")
]

WIN10_FEATURES = [
    ("cpu_atom_branch-load-misses_.csv", "cpu_atom_branch-load-misses_std", "std"),
    ("cpu_atom_cpu-cycles_.csv", "cpu_atom_cpu-cycles_max", "max"),
    ("cpu_core_cache-misses_.csv", "cpu_core_cache-misses_std", "std"),
    ("cpu_atom_cache-references_.csv", "cpu_atom_cache-references_max", "max")
]


# ======================== 统计量计算 ========================
def calc_statistic(values, stat_type):
    if stat_type == "mean":
        return np.mean(values)
    elif stat_type == "std":
        return np.std(values, ddof=1)
    elif stat_type == "med":
        return np.median(values)
    elif stat_type == "max":
        return np.max(values)
    else:
        return np.mean(values)


# ======================== 统一阈值计算 ========================
def calc_threshold_unified(feature_list, phase_name, perturbation=1.0):
    """
    统一使用 2.5%~97.5% 分位数（所有平台/特征相同）
    """
    data = np.array(feature_list, dtype=float)

    p2_5 = np.percentile(data, 2.5) * perturbation
    p97_5 = np.percentile(data, 97.5) * perturbation
    median_val = np.median(data)

    if phase_name == "start":
        low = 0.5 * median_val * perturbation
        high = p97_5
    elif phase_name == "key":
        low = p2_5
        high = p97_5
    elif phase_name == "encrypt":
        low = 0.5 * median_val * perturbation
        high = p97_5
    else:
        low, high = 0, 0

    return round(low, 4), round(high, 4)


# ======================== 计算对齐准确率 ========================
def compute_alignment_accuracy(df, stat_type, phase_cols, thresholds):
    low, high = thresholds
    feature_vals = []

    for _, row in df.iterrows():
        phase_vals = row[phase_cols].values.astype(float)
        feat_val = calc_statistic(phase_vals, stat_type)
        feature_vals.append(feat_val)

    if low is None or high is None or len(feature_vals) == 0:
        return 0.0

    within_range = sum(1 for v in feature_vals if low <= v <= high)
    return round(within_range / len(feature_vals) * 100, 2)


# ======================== 平台处理 ========================
def process_platform(platform, base_path, feature_configs):
    results = []
    robustness_results = []

    for fname, feature_name, stat_type in feature_configs:
        file_path = os.path.join(base_path, fname)

        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue

        df = pd.read_csv(file_path, encoding='latin-1')

        benign_samples = df[df["label"] == 0]
        malicious_samples = df[df["label"] == 1]

        if len(benign_samples) == 0 or len(malicious_samples) == 0:
            print(f"⚠️ {fname} 样本不足")
            continue

        start_cols = [f"data_{i}" for i in PHASE_START]
        key_cols = [f"data_{i}" for i in PHASE_KEY]
        encrypt_cols = [f"data_{i}" for i in PHASE_ENCRYPT]

        start_vals = []
        key_vals = []
        encrypt_vals = []

        for _, row in benign_samples.iterrows():
            start_vals.append(calc_statistic(row[start_cols].values.astype(float), stat_type))
            key_vals.append(calc_statistic(row[key_cols].values.astype(float), stat_type))
            encrypt_vals.append(calc_statistic(row[encrypt_cols].values.astype(float), stat_type))

        t_start_orig = calc_threshold_unified(start_vals, "start", 1.0)
        t_key_orig = calc_threshold_unified(key_vals, "key", 1.0)
        t_enc_orig = calc_threshold_unified(encrypt_vals, "encrypt", 1.0)

        perturbations = [0.8, 1.2]
        phase_configs = [
            ("start", start_cols, t_start_orig, start_vals),
            ("key", key_cols, t_key_orig, key_vals),
            ("encrypt", encrypt_cols, t_enc_orig, encrypt_vals)
        ]

        row_robustness = {"feature": feature_name, "platform": platform}

        for phase_name, cols, orig_thresholds, phase_vals in phase_configs:
            orig_acc = compute_alignment_accuracy(malicious_samples, stat_type, cols, orig_thresholds)
            row_robustness[f"{phase_name}_orig"] = orig_acc

            for pert in perturbations:
                t_low, t_high = calc_threshold_unified(phase_vals, phase_name, pert)
                pert_acc = compute_alignment_accuracy(malicious_samples, stat_type, cols, (t_low, t_high))
                change = round(pert_acc - orig_acc, 2)
                row_robustness[f"{phase_name}_pert{int(pert * 100)}"] = pert_acc
                row_robustness[f"{phase_name}_change{int(pert * 100)}"] = change

        robustness_results.append(row_robustness)

        results.append({
            "platform": platform,
            "feature": feature_name,
            "stat": stat_type,
            "start_low": t_start_orig[0],
            "start_high": t_start_orig[1],
            "key_low": t_key_orig[0],
            "key_high": t_key_orig[1],
            "encrypt_low": t_enc_orig[0],
            "encrypt_high": t_enc_orig[1]
        })

        print(f"✅ {platform}: {feature_name} ({stat_type}) 完成")

    return results, robustness_results


# ======================== 执行 ========================
print("=" * 60)
print("阶段阈值计算 + 扰动鲁棒性分析 (±20%)")
print("阈值方法: 统一规则 (2.5%~97.5% 分位数)")
print(f"输出目录: {OUTPUT_DIR}")
print("=" * 60)

win7_results, win7_robustness = process_platform("Win7", base_win7, WIN7_FEATURES)
win10_results, win10_robustness = process_platform("Win10", base_win10, WIN10_FEATURES)

all_results = win7_results + win10_results
df_thresholds = pd.DataFrame(all_results)
thresholds_path = os.path.join(OUTPUT_DIR, "thresholds_unified.csv")
df_thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")

all_robustness = win7_robustness + win10_robustness
df_robustness = pd.DataFrame(all_robustness)
robustness_path = os.path.join(OUTPUT_DIR, "robustness_unified.csv")
df_robustness.to_csv(robustness_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("扰动鲁棒性分析汇总 (统一阈值)")
print("=" * 60)

robust_summary = []

for _, row in df_robustness.iterrows():
    print(f"\n{row['platform']}: {row['feature']}")

    max_change = 0
    for phase in ["start", "key", "encrypt"]:
        for pert in [80, 120]:
            change_col = f"{phase}_change{pert}"
            if change_col in row:
                change = abs(row[change_col])
                max_change = max(max_change, change)
                print(f"  {phase} ±20%: 变化 {row[change_col]:+.2f}%")

    status = "Robust" if max_change < 10 else "Sensitive"
    print(f"  {status} (max change: {max_change:.2f}%)")

    robust_summary.append({
        "platform": row['platform'],
        "feature": row['feature'],
        "max_change_pct": max_change,
        "status": status
    })

df_summary = pd.DataFrame(robust_summary)
summary_path = os.path.join(OUTPUT_DIR, "robustness_summary_unified.csv")
df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("✅ 计算完成！")
print(f"📁 阈值文件: {thresholds_path}")
print(f"📁 鲁棒性分析: {robustness_path}")
print(f"📁 鲁棒性汇总: {summary_path}")
print("=" * 60)

print("\n鲁棒性汇总统计:")
print(df_summary['status'].value_counts())
print(f"平均最大变化: {df_summary['max_change_pct'].mean():.2f}%")