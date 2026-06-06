import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================== 路径配置 ========================
base_win7 = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win7"
base_win10 = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata\win10"
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\win7_win10_3phasedata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== 阶段划分 ========================
PHASE_START = list(range(1, 21))  # 0-2s
PHASE_KEY = list(range(21, 51))  # 2-5s
PHASE_ENCRYPT = list(range(51, 100))  # 5-10s
PHASE_TOTAL = list(range(1, 100))  # 0-10s

# ======================== 特征配置（基于最新筛选结果） ========================
# Win7 特征: (文件名, 特征名, 统计量)
WIN7_FEATURES = [
    ("node-stores.csv", "node-stores_mean", "mean"),
    ("cache-misses.csv", "cache-misses_std", "std"),
    ("cpu-cycles.csv", "cpu-cycles_med", "med"),
    ("node-stores.csv", "node-stores_std", "std")
]

# Win10 特征: (文件名, 特征名, 统计量)
WIN10_FEATURES = [
    ("cpu_atom_branch-load-misses_.csv", "cpu_atom_branch-load-misses_std", "std"),
    ("cpu_atom_cpu-cycles_.csv", "cpu_atom_cpu-cycles_max", "max"),
    ("cpu_core_cache-misses_.csv", "cpu_core_cache-misses_std", "std"),
    ("cpu_atom_cache-references_.csv", "cpu_atom_cache-references_max", "max")
]

# ======================== 阶段权重（基于阶段对齐得分） ========================
# 格式: 特征名 -> [startup_score, key_score, encryption_score]
# 注意：这些分数需要根据您的实际阶段对齐结果更新
FEATURE_WEIGHTS = {
    # Win7
    "node-stores_mean": [58.26, 91.45, 82.24],
    "cache-misses_std": [69.47, 65.44, 80.04],
    "cpu-cycles_med": [54.61, 72.98, 73.58],
    "node-stores_std": [58.26, 91.45, 82.24],
    # Win10
    "cpu_atom_branch-load-misses_std": [84.90, 49.59, 94.87],
    "cpu_atom_cpu-cycles_max": [100.00, 49.04, 100.00],
    "cpu_core_cache-misses_std": [89.88, 76.76, 100.00],
    "cpu_atom_cache-references_max": [90.48, 66.63, 100.00]
}


# ======================== 统计量计算函数 ========================
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


# ======================== 特征值计算 ========================
def compute_feature_value(filename, stat_type, phase_values):
    if stat_type == "mean":
        return np.mean(phase_values)
    elif stat_type == "std":
        return np.std(phase_values, ddof=1)
    elif stat_type == "med":
        return np.median(phase_values)
    elif stat_type == "max":
        return np.max(phase_values)
    else:
        return np.mean(phase_values)


# ======================== 加载阈值 ========================
df_th = pd.read_csv(os.path.join(OUTPUT_DIR, "thresholds_unified.csv"))
THRESHOLD = {}
for _, row in df_th.iterrows():
    platform = row["platform"]
    feature = row["feature"]
    THRESHOLD[(platform, feature)] = {
        "start": (row["start_low"], row["start_high"]),
        "key": (row["key_low"], row["key_high"]),
        "encrypt": (row["encrypt_low"], row["encrypt_high"])
    }


# ======================== 评估函数 ========================
def evaluate_platform(platform, base_path, feature_configs):
    # 收集所有样本数据
    sample_data = {}

    for fname, feature_name, stat_type in feature_configs:
        file_path = os.path.join(base_path, fname)
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue

        df = pd.read_csv(file_path, encoding='latin-1')

        for idx, row in df.iterrows():
            if idx not in sample_data:
                sample_data[idx] = {"label": row["label"], "feats": {}}

            # 计算各阶段特征值
            start_vals = row[[f"data_{i}" for i in PHASE_START]].values.astype(float)
            key_vals = row[[f"data_{i}" for i in PHASE_KEY]].values.astype(float)
            encrypt_vals = row[[f"data_{i}" for i in PHASE_ENCRYPT]].values.astype(float)

            feat_start = compute_feature_value(fname, stat_type, start_vals)
            feat_key = compute_feature_value(fname, stat_type, key_vals)
            feat_encrypt = compute_feature_value(fname, stat_type, encrypt_vals)

            sample_data[idx]["feats"][feature_name] = {
                "start": feat_start,
                "key": feat_key,
                "encrypt": feat_encrypt
            }

    # 计算每个样本的加权违规分数
    y_true = []
    start_scores = []
    key_scores = []
    encrypt_scores = []

    for idx in sample_data:
        label = sample_data[idx]["label"]
        y_true.append(label)

        s_sum = 0.0
        k_sum = 0.0
        e_sum = 0.0

        for feature_name, feats in sample_data[idx]["feats"].items():
            if (platform, feature_name) not in THRESHOLD:
                continue

            th = THRESHOLD[(platform, feature_name)]
            weights = FEATURE_WEIGHTS.get(feature_name, [50, 50, 50])

            # 检查是否超出阈值（超出则累加权重）
            if feats["start"] < th["start"][0] or feats["start"] > th["start"][1]:
                s_sum += weights[0]
            if feats["key"] < th["key"][0] or feats["key"] > th["key"][1]:
                k_sum += weights[1]
            if feats["encrypt"] < th["encrypt"][0] or feats["encrypt"] > th["encrypt"][1]:
                e_sum += weights[2]

        start_scores.append(s_sum)
        key_scores.append(k_sum)
        encrypt_scores.append(e_sum)

    # 根据百分位数确定阈值并预测
    def predict(scores, percentile=80):
        th = np.percentile(scores, percentile)
        return [1 if s >= th else 0 for s in scores]

    def calculate_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        return [round(fpr, 2), round(prec, 2), round(rec, 2), round(f1, 2), round(acc, 2)]

    # 各阶段预测（使用不同的百分位阈值）
    y_pred_start = predict(start_scores, percentile=80)
    y_pred_key = predict(key_scores, percentile=80)
    y_pred_encrypt = predict(encrypt_scores, percentile=80)

    # 综合预测：任一阶段违规即判定为恶意
    y_pred_ensemble = [1 if (s > 0 or k > 0 or e > 0) else 0
                       for s, k, e in zip(start_scores, key_scores, encrypt_scores)]

    metrics_start = calculate_metrics(y_true, y_pred_start)
    metrics_key = calculate_metrics(y_true, y_pred_key)
    metrics_encrypt = calculate_metrics(y_true, y_pred_encrypt)
    metrics_ensemble = calculate_metrics(y_true, y_pred_ensemble)

    return metrics_start, metrics_key, metrics_encrypt, metrics_ensemble


# ======================== 执行 ========================
print("=" * 60)
print("离线回溯验证")
print("=" * 60)

print("\n处理 Win7...")
ms7, mk7, me7, m_ens7 = evaluate_platform("Win7", base_win7, WIN7_FEATURES)

print("\n处理 Win10...")
ms10, mk10, me10, m_ens10 = evaluate_platform("Win10", base_win10, WIN10_FEATURES)

# ======================== 输出结果 ========================
cols = ["阶段名称", "误报率(%)", "精确率(%)", "召回率(%)", "F1(%)", "准确率(%)"]

df7 = pd.DataFrame([
    ["启动(0-2s)", *ms7],
    ["密钥(2-5s)", *mk7],
    ["加密(5-10s)", *me7],
    ["综合(任一阶段)", *m_ens7]
], columns=cols)

df10 = pd.DataFrame([
    ["启动(0-2s)", *ms10],
    ["密钥(2-5s)", *mk10],
    ["加密(5-10s)", *me10],
    ["综合(任一阶段)", *m_ens10]
], columns=cols)

# 保存结果
df7.to_csv(os.path.join(OUTPUT_DIR, "result_win7_offline.csv"), index=False, encoding="utf-8-sig")
df10.to_csv(os.path.join(OUTPUT_DIR, "result_win10_offline.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("=============== Win7 离线回溯验证结果 ===============")
print(df7.to_string(index=False))

print("\n=============== Win10 离线回溯验证结果 ===============")
print(df10.to_string(index=False))
print("=" * 60)

print(f"\n✅ 结果已保存至: {OUTPUT_DIR}")