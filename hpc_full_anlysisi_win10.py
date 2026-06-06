"""
HPC 完整分析 + 样本级特征矩阵（一行一样本）
✅ 使用统一后的共有样本数据（两个平台均已过滤）
✅ 新增 FDR 多重比较校正
✅ 新增 Cohen's d 效应量分级
✅ 输出事件列表供附录使用
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import pointbiserialr
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 请修改此处为您的统一后的数据文件夹 ======================
CSV_FOLDER = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\merged_result_win10"   # 统一后的 Win10 数据文件夹
SAVE_DIR   = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10"     # 输出目录
ENCODING   = "latin-1"
os.makedirs(SAVE_DIR, exist_ok=True)
# ============================================================================

# ---------------------- 加载所有事件 ----------------------
def load_all_events(folder):
    events = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        try:
            df = pd.read_csv(path, encoding=ENCODING)
        except:
            continue
        if "label" not in df.columns or len(df) < 10:
            continue
        data_cols = [c for c in df.columns if c.startswith("data_")]
        if not data_cols:
            continue
        event_name = os.path.splitext(fname)[0]
        events[event_name] = df
    return events

events = load_all_events(CSV_FOLDER)

# ---------------------- 安全计算函数 ----------------------
def safe_mean(arr):
    arr = arr[np.isfinite(arr)]
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_correlation(y, x):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10 or np.std(x) < 1e-6:
        return 0.0, 1.0
    try:
        return pointbiserialr(y, x)
    except:
        return 0.0, 1.0

def cohen_d(x0, x1):
    n0, n1 = len(x0), len(x1)
    s0 = np.var(x0, ddof=1) if len(x0) > 1 else 0.0
    s1 = np.var(x1, ddof=1) if len(x1) > 1 else 0.0
    pooled = np.sqrt(((n0-1)*s0 + (n1-1)*s1) / (n0+n1-2 + 1e-9))
    return (np.mean(x1)-np.mean(x0)) / (pooled + 1e-9)

def level_correlation(abs_r):
    if abs_r >= 0.6: return "强相关"
    elif abs_r >= 0.4: return "中等相关"
    else: return "弱相关"

def level_significance(p):
    if p < 1e-10: return "极显著"
    elif p < 0.05: return "显著"
    else: return "不显著"

def level_fold_change(fc):
    if fc >= 2.0: return "高差异"
    elif fc >= 1.5: return "中等差异"
    else: return "低差异"

def level_cohen_d(d):
    """Cohen's d 效应量分级（绝对值）"""
    abs_d = abs(d)
    if abs_d >= 0.8: return "大效应"
    elif abs_d >= 0.5: return "中等效应"
    else: return "小效应"

# ---------------------- 完整分析：使用真实相关系数 r ----------------------
rows = []
all_feature_vectors = {}

for evt, df in events.items():
    data_cols = [c for c in df.columns if c.startswith("data_")]
    X = df[data_cols].values
    y = df["label"].values

    mean_v = np.nanmean(X, axis=1)
    std_v  = np.nanstd(X, axis=1, ddof=1)
    max_v  = np.nanmax(X, axis=1)
    med_v  = np.nanmedian(X, axis=1)

    for name, vec in [("mean", mean_v), ("std", std_v), ("max", max_v), ("med", med_v)]:
        vec0 = vec[y == 0]
        vec1 = vec[y == 1]
        mu0 = safe_mean(vec0)
        mu1 = safe_mean(vec1)

        if mu0 < 1e-6:
            fc = 1.0
            growth = 0.0
        else:
            fc = mu1 / mu0
            growth = ((mu1 - mu0) / mu0) * 100

        r, p = safe_correlation(y, vec)
        abs_r = abs(r)
        d = cohen_d(vec0, vec1)

        feat_name = f"{evt}_{name}"
        all_feature_vectors[feat_name] = vec

        rows.append({
            "event": evt,
            "stat": name,
            "feature": feat_name,
            "corr_r": round(float(r), 4),       # 真实相关系数（可负）
            "abs_r": round(abs_r, 4),
            "corr_level": level_correlation(abs_r),
            "p_value": round(float(p), 6),
            "sig_level": level_significance(p),
            "fold_change": round(float(fc), 4),
            "fc_level": level_fold_change(fc),
            "growth_pct": round(float(growth), 4),
            "mean_benign": round(float(mu0), 4),
            "mean_malicious": round(float(mu1), 4),
            "cohen_d": round(float(d), 4),
            "cohen_d_level": level_cohen_d(d)
        })

# ====================== 输出1：完整结果（含 FDR 校正） ======================
df_out = pd.DataFrame(rows)

# FDR 多重比较校正（Benjamini-Hochberg）
reject, pvals_corrected = fdrcorrection(df_out['p_value'].values, alpha=0.05)
df_out['p_fdr'] = pvals_corrected
df_out['sig_fdr'] = ['显著' if r else '不显著' for r in reject]

# 保存完整结果
df_out.to_csv(os.path.join(SAVE_DIR, "HPC_42events_FINAL_with_FDR.csv"), index=False, encoding="utf-8-sig")

# ====================== 输出2：事件列表（供附录） ======================
event_list = sorted(set(df_out['event']))
pd.Series(event_list).to_csv(os.path.join(SAVE_DIR, "HPC_event_list.csv"), index=False, header=False, encoding="utf-8-sig")
print(f"共 {len(event_list)} 个事件/指标")

# ====================== 输出3：特征级相关性矩阵 ======================
f_names = list(all_feature_vectors.keys())
vec_list = [all_feature_vectors[f] for f in f_names]
corr_mat = pd.DataFrame(np.corrcoef(vec_list), index=f_names, columns=f_names)
corr_mat.to_csv(os.path.join(SAVE_DIR, "HPC_feature_correlation_matrix.csv"), encoding="utf-8-sig")

# ====================== 输出4：样本特征矩阵 ======================
feature_dict = {}
for evt, df in events.items():
    data_cols = [c for c in df.columns if c.startswith("data_")]
    X = df[data_cols].values

    m = np.nanmean(X, axis=1)
    s = np.nanstd(X, axis=1, ddof=1)
    mx = np.nanmax(X, axis=1)
    med = np.nanmedian(X, axis=1)

    feature_dict[f"{evt}_mean"] = m
    feature_dict[f"{evt}_std"] = s
    feature_dict[f"{evt}_max"] = mx
    feature_dict[f"{evt}_median"] = med

df_feature = pd.DataFrame(feature_dict)
any_df = next(iter(events.values()))
df_feature["label"] = any_df["label"].values

df_feature.to_csv(os.path.join(SAVE_DIR, "HPC_sample_feature_matrix.csv"), index=False, encoding="utf-8-sig")

# ====================== 绘图：使用真实相关系数 ======================
df_best = df_out.loc[df_out.groupby("event")["abs_r"].idxmax()]

# 散点图
plt.figure(figsize=(12,7))
plt.scatter(df_best["corr_r"], df_best["fold_change"], s=120, alpha=0.7)
for _, row in df_best.iterrows():
    plt.annotate(row["event"], (row["corr_r"], row["fold_change"]), fontsize=8)
plt.axvline(0.6, c='r', ls='--', label="r≥0.6 强正相关")
plt.axvline(-0.6, c='r', ls='--', label="r≤-0.6 强负相关")
plt.axvline(0, c='black', lw=1)
plt.axhline(2, c='g', ls='--', label="FC≥2 高差异")
plt.xlabel("Correlation Coefficient (r)")
plt.ylabel("Fold Change")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_vs_fc.png"), dpi=300)
plt.close()

# Top10 相关性（按绝对值）
top10 = df_best.reindex(df_best["abs_r"].abs().nlargest(10).index)
plt.figure(figsize=(12,6))
bars = plt.bar(top10["event"], top10["corr_r"], color=['#e74c3c' if x < 0 else '#2ecc71' for x in top10["corr_r"]])
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=1)
plt.title("Top 10 HPC Events with the Strongest Correlation to Ransomware Labels")
plt.ylabel("Correlation Coefficient (r)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "top10_corr.png"), dpi=300)
plt.close()

# 热力图
plt.figure(figsize=(22, 18))
sns.heatmap(corr_mat, cmap="RdBu_r", center=0, linewidths=0.2, annot=False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "feature_corr_heatmap.png"), dpi=300)
plt.close()

print("🎉 运行完成！")
print(f"✅ 完整结果（含 FDR）：{os.path.join(SAVE_DIR, 'HPC_42events_FINAL_with_FDR.csv')}")
print(f"✅ 事件列表：{os.path.join(SAVE_DIR, 'HPC_event_list.csv')}")
print(f"✅ 样本特征矩阵：{os.path.join(SAVE_DIR, 'HPC_sample_feature_matrix.csv')}")