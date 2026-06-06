import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== 路径（完全沿用你的） ======================
base_path = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10"

# 1. 相关矩阵（方阵）
win7_corr_matrix = os.path.join(base_path, "redundancy_correlation_matrix.csv")
win10_corr_matrix = os.path.join(base_path, "HPC_feature_correlation_matrix.csv")

# 2. 相关性 & 差异倍数分析表
win7_stat_path = os.path.join(base_path, "HPC_25events_FINAL_ANALYSIS.csv")
win10_stat_path = os.path.join(base_path, "HPC_42events_FINAL_with_FDR.csv")

out_dir = os.path.join(base_path, "HPC冗余分析_全套结果_英文图例版")
os.makedirs(out_dir, exist_ok=True)

# ====================== 工具函数 ======================
def clean(s):
    return str(s).strip().rstrip("_").replace(" ", "")

def corr_matrix_to_pairs(mat_path):
    df = pd.read_csv(mat_path, index_col=0)
    df.index = [clean(x) for x in df.index]
    df.columns = [clean(x) for x in df.columns]
    pairs = []
    events = df.columns.tolist()
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            e1 = events[i]
            e2 = events[j]
            val = df.iloc[i, j]
            if pd.isna(val):
                continue
            pairs.append([e1, e2, abs(val)])
    return pd.DataFrame(pairs, columns=["事件1", "事件2", "冗余系数"]), events, df

# ====================== 1. 加载数据 ======================
df7_red, events7, mat7 = corr_matrix_to_pairs(win7_corr_matrix)
df10_red, events10, mat10 = corr_matrix_to_pairs(win10_corr_matrix)

df7_stat = pd.read_csv(win7_stat_path)
df10_stat = pd.read_csv(win10_stat_path)

# 数值清洗
df7_stat["abs_r"] = pd.to_numeric(df7_stat["abs_r"], errors="coerce")
df10_stat["abs_r"] = pd.to_numeric(df10_stat["abs_r"], errors="coerce")
df7_stat["fold_change"] = pd.to_numeric(df7_stat["fold_change"], errors="coerce")
df10_stat["fold_change"] = pd.to_numeric(df10_stat["fold_change"], errors="coerce")

# 统一特征名（修复匹配问题）
df7_stat["feat"] = df7_stat["event"].apply(clean)
df10_stat["feat"] = df10_stat["event"].apply(clean)

# ====================== 2. 高冗余事件对 ======================
THRESH_RED = 0.8
high7 = df7_red[df7_red["冗余系数"] >= THRESH_RED].copy()
high10 = df10_red[df10_red["冗余系数"] >= THRESH_RED].copy()

high7.to_csv(os.path.join(out_dir, "Win7_High_Redundancy_Events.csv"), index=False, encoding="utf-8-sig")
high10.to_csv(os.path.join(out_dir, "Win10_High_Redundancy_Events.csv"), index=False, encoding="utf-8-sig")

# 获取冗余特征集合
red7_feat = set(high7["事件1"]) | set(high7["事件2"])
red10_feat = set(high10["事件1"]) | set(high10["事件2"])

# ====================== 3. 低冗余优质特征（已修复！） ======================
def get_good_features(df_stat, redundant_feats):
    good = df_stat[
        (~df_stat["feat"].isin(redundant_feats)) &
        (df_stat["abs_r"] >= 0.1) &
        (df_stat["fold_change"] >= 1.1)
    ].copy()
    return good

good7 = get_good_features(df7_stat, red7_feat)
good10 = get_good_features(df10_stat, red10_feat)

good7.to_csv(os.path.join(out_dir, "Win7_LowRedundancy_GoodFeatures.csv"), index=False, encoding="utf-8-sig")
good10.to_csv(os.path.join(out_dir, "Win10_LowRedundancy_GoodFeatures.csv"), index=False, encoding="utf-8-sig")

# ====================== 4. 冗余系数分布直方图 ======================
plt.figure(figsize=(12,6))
plt.hist(df7_red["冗余系数"], bins=50, alpha=0.7, color="#1f77b4", label="Win7")
plt.hist(df10_red["冗余系数"], bins=50, alpha=0.7, color="#ff7f0e", label="Win10")
plt.xlabel("Redundancy Coefficient |r|")
plt.ylabel("Frequency")
plt.title("Redundancy Coefficient Distribution: Win7 vs Win10")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(out_dir, "Redundancy_Coefficient_Distribution.png"), bbox_inches="tight", dpi=300)
plt.close()

# ====================== 5. 热力图 ======================
def heatmap(mat, title, savepath):
    plt.figure(figsize=(16,14))
    im = plt.imshow(mat.abs(), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, shrink=0.8, label="Redundancy Coefficient |r|")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", dpi=300)
    plt.close()

heatmap(mat7, "Win7 HPC Redundancy Heatmap", os.path.join(out_dir, "Win7_Heatmap.png"))
heatmap(mat10, "Win10 HPC Redundancy Heatmap", os.path.join(out_dir, "Win10_Heatmap.png"))

# ====================== 6. 优质特征散点图 ======================
plt.figure(figsize=(12,6))
if not good7.empty:
    plt.scatter(good7["abs_r"], good7["fold_change"], c="#1f77b4", label="Win7")
if not good10.empty:
    plt.scatter(good10["abs_r"], good10["fold_change"], c="#ff7f0e", label="Win10")
plt.xlabel("abs_r (Correlation with Label)")
plt.ylabel("fold_change (Difference Multiple)")
plt.title("Low-Redundancy High-Discrimination Features")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(out_dir, "Low_Redundancy_Features_Scatter.png"), bbox_inches="tight", dpi=300)
plt.close()

# ====================== 7. 高冗余数量柱状图 ======================
plt.figure(figsize=(10,5))
plt.bar(["Win7","Win10"], [len(high7), len(high10)], color=["#1f77b4","#ff7f0e"])
plt.title(f"High Redundancy Pairs (r≥{THRESH_RED})")
plt.ylabel("Count")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "High_Redundancy_Count.png"), bbox_inches="tight", dpi=300)
plt.close()

# ====================== 8. 完整分析报告 ======================
report = f"""
HPC 冗余特征综合分析报告
=============================
一、数据概况
- Win7 事件数：{len(events7)} 个
- Win10 事件数：{len(events10)} 个
- Win7 事件对总数：{len(df7_red)} 对
- Win10 事件对总数：{len(df10_red)} 对

二、高冗余事件对(|r|≥{THRESH_RED})
- Win7：{len(high7)} 对
- Win10：{len(high10)} 对

三、平均冗余系数
- Win7：{df7_red['冗余系数'].mean():.3f}
- Win10：{df10_red['冗余系数'].mean():.3f}

四、低冗余优质特征
- Win7：{len(good7)} 个
- Win10：{len(good10)} 个

筛选条件：
1. 不在高冗余清单中
2. abs_r ≥ 0.1
3. fold_change ≥ 1.1
"""

with open(os.path.join(out_dir, "冗余分析总结报告.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# ====================== 完成 ======================
print("✅ 全套分析结果已全部生成！")
print(f"📁 路径：{out_dir}")
print(f"✅ Win7 优质低冗余特征：{len(good7)} 个")
print(f"✅ Win10 优质低冗余特征：{len(good10)} 个")