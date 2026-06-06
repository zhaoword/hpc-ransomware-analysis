import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ====================== 路径配置 ======================
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\Initial_Feature_Screening"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIN7_CSV = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\redundancy\HPC_25events_FINAL_ANALYSIS.csv"
WIN10_CSV = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\redundancy\HPC_42events_FINAL_with_FDR.csv"

# ====================== 阈值配置 ======================
ABS_R_THRESH = 0.20
FC_THRESH_LO = 1 / 1.3  # 约 0.769
FC_THRESH_HI = 1.3
COHEN_D_THRESH = 0.2
SIG_LEVEL = ["Significant", "显著"]


# ====================== 初筛 ======================
def filter_features(df, platform):
    if platform == "win7":
        sig_mask = df["sig_fdr"] == "Significant"
    else:
        sig_mask = df["sig_fdr"] == "显著"

    return df[
        (df["abs_r"].abs() >= ABS_R_THRESH) &
        sig_mask &
        ((df["fold_change"] >= FC_THRESH_HI) | (df["fold_change"] <= FC_THRESH_LO)) &
        (df["cohen_d"].abs() >= COHEN_D_THRESH)
        ].copy()


# ====================== 按相关性排序，取 Top-K ======================
def select_top_k(df, k):
    df = df.sort_values("abs_r", ascending=False).head(k)
    return df.sort_values("abs_r", ascending=True)  # 用于绘图


# ====================== 加载并筛选 ======================
df7 = pd.read_csv(WIN7_CSV)
df10 = pd.read_csv(WIN10_CSV)

f7 = filter_features(df7, "win7")
f10 = filter_features(df10, "win10")

# 取 Top-K（Win7 取 9 个，Win10 取 14 个）
K7 = 9
K10 = 14
f7 = select_top_k(f7, K7)
f10 = select_top_k(f10, K10)

print(f"Win7 筛选结果：{len(f7)} 个特征")
print(f"Win10 筛选结果：{len(f10)} 个特征")

# ====================== 保存结果 ======================
f7.to_csv(os.path.join(OUTPUT_DIR, "Win7_Filtered_Features.csv"), index=False, encoding="utf-8-sig")
f10.to_csv(os.path.join(OUTPUT_DIR, "Win10_Filtered_Features.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("📊 Windows 7 最终筛选特征")
print("=" * 60)
print(f7[["feature", "abs_r", "sig_fdr", "fold_change", "cohen_d"]].to_string(index=False))

print("\n" + "=" * 60)
print("📊 Windows 10 最终筛选特征")
print("=" * 60)
print(f10[["feature", "abs_r", "sig_fdr", "fold_change", "cohen_d"]].to_string(index=False))

# ====================== 绘图 ======================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 10))

# Win7
if len(f7) > 0:
    ax1.barh(f7["feature"], f7["abs_r"], color='#3366cc', height=0.6)
    ax1.set_xlim(0, 0.5)
else:
    ax1.text(0.5, 0.5, 'No features selected', ha='center', va='center')
ax1.set_xlabel('Absolute Correlation (abs_r)', weight='bold', fontsize=9)
ax1.set_title('Windows 7 (i7-7500U)\nFiltered HPC Features', weight='bold', fontsize=10)
ax1.grid(axis='x', linestyle=':', alpha=0.4)
ax1.tick_params(axis='y', labelsize=7)

# Win10
if len(f10) > 0:
    ax2.barh(f10["feature"], f10["abs_r"], color='#cc3333', height=0.6)
    ax2.set_xlim(0, 0.9)
else:
    ax2.text(0.5, 0.5, 'No features selected', ha='center', va='center')
ax2.set_xlabel('Absolute Correlation (abs_r)', weight='bold', fontsize=9)
ax2.set_title('Windows 10 (Ultra9 185H)\nFiltered HPC Features', weight='bold', fontsize=10)
ax2.grid(axis='x', linestyle=':', alpha=0.4)
ax2.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cross_platform_filtered_hpc.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'cross_platform_filtered_hpc.pdf'), bbox_inches='tight')
plt.show()

print(f"\n✅ 图片保存至：{os.path.join(OUTPUT_DIR, 'cross_platform_filtered_hpc.png')}")