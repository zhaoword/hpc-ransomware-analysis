import os
import pandas as pd
import numpy as np

# ====================== 路径配置 ======================
WIN7_PATH = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win7"
WIN10_PATH = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win10"
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 三段时间划分 ======================
PHASE_SLICES = {
    "0-2s": slice(0, 20),
    "2-5s": slice(20, 50),
    "5-10s": slice(50, 99)
}

# ====================== 特征计算规则（基于聚类筛选结果） ======================
WIN7_FEATURES_LIST = [
    ("node-stores.csv", "mean"),
    ("node-stores.csv", "std"),
    ("cache-misses.csv", "std"),
    ("cpu-cycles.csv", "med")
]

WIN10_FEATURES_LIST = [
    ("cpu_atom_branch-load-misses_.csv", "std"),
    ("cpu_atom_cpu-cycles_.csv", "max"),
    ("cpu_core_cache-misses_.csv", "std"),
    ("cpu_atom_cache-references_.csv", "max")
]


# ====================== 统计量计算 ======================
def calc_stat(series, stat_type):
    if stat_type == "mean":
        return np.mean(series)
    elif stat_type == "std":
        return np.std(series, ddof=1)
    elif stat_type == "med":
        return np.median(series)
    elif stat_type == "max":
        return np.max(series)
    return 0


# ====================== 三个基础指标 ======================
def compute_3_indicators(segment_data):
    std_val = np.std(segment_data, ddof=1)
    mean_val = np.mean(segment_data)
    peak_val = np.max(segment_data)
    peak_sig = peak_val / mean_val if mean_val != 0 else 0
    return std_val, peak_sig, mean_val


# ====================== 处理单个文件 ======================
def process_file(file, path, stat_type):
    df = pd.read_csv(os.path.join(path, file), encoding='latin-1')
    data_cols = [f"data_{i}" for i in range(1, 100)]
    labels = df["label"].values
    results = []

    for idx in range(len(df)):
        row = df.iloc[idx][data_cols].values.astype(float)
        for phase_name, slc in PHASE_SLICES.items():
            seg = row[slc]
            feat_val = calc_stat(seg, stat_type)
            std_val, peak_sig, mean_val = compute_3_indicators(seg)

            results.append({
                "platform": "Win7" if path == WIN7_PATH else "Win10",
                "event_file": file,
                "stat": stat_type,
                "sample": idx,
                "label": labels[idx],
                "phase": phase_name,
                "your_feature": feat_val,
                "fluctuation_std": round(std_val, 4),
                "peak_significance": round(peak_sig, 4),
                "phase_mean": round(mean_val, 4)
            })
    return pd.DataFrame(results)


# ====================== 主流程 ======================
if __name__ == "__main__":
    all_results = []

    # 处理 Win7
    for f, t in WIN7_FEATURES_LIST:
        print(f"处理 Win7: {f} -> {t}")
        df = process_file(f, WIN7_PATH, t)
        all_results.append(df)

    # 处理 Win10
    for f, t in WIN10_FEATURES_LIST:
        print(f"处理 Win10: {f} -> {t}")
        df = process_file(f, WIN10_PATH, t)
        all_results.append(df)

    df_total = pd.concat(all_results)

    # ====================== 计算 恶意-良性差异度 ======================
    diff_list = []
    grouped = df_total.groupby(["event_file", "stat", "phase", "label"])["phase_mean"].mean().reset_index()

    for (evt, stat, phase), g in grouped.groupby(["event_file", "stat", "phase"]):
        mal = g[g["label"] == 1]["phase_mean"].values
        ben = g[g["label"] == 0]["phase_mean"].values
        mal_mean = mal[0] if len(mal) > 0 else 0
        ben_mean = ben[0] if len(ben) > 0 else 0
        diff = abs(mal_mean - ben_mean) / (ben_mean + 1e-8)
        diff_list.append({
            "event_file": evt,
            "stat": stat,
            "phase": phase,
            "mal_benign_diff": round(diff, 4)
        })

    diff_df = pd.DataFrame(diff_list)
    df_total = df_total.merge(diff_df, on=["event_file", "stat", "phase"])


    # ====================== 百分制打分 ======================
    def score_row(std, peak, diff):
        score_std = min(30, std * 3)
        score_peak = min(35, peak * 7)
        score_diff = min(35, diff * 100)
        return round(score_std + score_peak + score_diff, 2)


    df_total["total_score"] = df_total.apply(
        lambda row: score_row(row["fluctuation_std"], row["peak_significance"], row["mal_benign_diff"]), axis=1
    )


    # ====================== S/A/B/C 分级 ======================
    def get_level(score):
        if score >= 85:
            return "S"
        elif score >= 70:
            return "A"
        elif score >= 50:
            return "B"
        else:
            return "C"


    df_total["level"] = df_total["total_score"].apply(get_level)

    # ====================== 保存最终结果 ======================
    output_path = os.path.join(OUTPUT_DIR, "final_phase_analysis.csv")
    output_cols = [
        "platform", "event_file", "stat", "phase", "label",
        "fluctuation_std", "peak_significance", "mal_benign_diff",
        "total_score", "level"
    ]
    final = df_total[output_cols].drop_duplicates(["event_file", "stat", "phase", "label"])
    final.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ====================== 打印汇总结果 ======================
    print("\n" + "=" * 60)
    print("阶段分析汇总结果")
    print("=" * 60)

    # 按平台和阶段统计 S/A/B/C 分布
    summary = final.groupby(["platform", "phase", "level"]).size().unstack(fill_value=0)
    print("\n各平台各阶段等级分布：")
    print(summary)

    # 按阶段打印平均得分
    print("\n各平台各阶段平均得分：")
    avg_score = final.groupby(["platform", "phase"])["total_score"].mean().round(2)
    print(avg_score)

    print(f"\n✅ 全部完成！结果已保存到：{output_path}")