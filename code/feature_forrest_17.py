import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
import os
warnings.filterwarnings('ignore')

# ---------------------- 1. 基础配置（图表样式+路径） ----------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10

# 数据路径与保存路径
data_path = "D:\\data_doctor\\win7_forrest\\statistical_features_25events.csv"
save_path = "D:\\data_doctor\\win7_forrest\\RF_Parameter_Optimization\\"
os.makedirs(save_path, exist_ok=True)  # 自动创建保存文件夹

# ---------------------- 2. 数据准备（✅ 改为 17 个 HPC 事件） ----------------------
# 2.1 读取数据
df = pd.read_csv(data_path)
print(f"原始数据形状：{df.shape}")

# 2.2 你指定的 17 个目标HPC事件
selected_events = [
    'branch-loads', 'branches', 'cache-references',
    'cpu-cycles', 'dTLB-load-misses', 'dTLB-loads',
    'dTLB-store-misses', 'dTLB-stores', 'instructions',
    'iTLB-load-misses', 'iTLB-loads', 'L1-dcache-load-misses',
    'L1-dcache-loads', 'L1-dcache-stores', 'L1-icache-load-misses',
    'LLC-loads', 'LLC-store'
]

# 2.3 提取特征列（17个事件的统计指标）
feature_cols = [
    col for col in df.columns
    if any(evt in col for evt in selected_events) and col != 'label'
]
X = df[feature_cols]
y = df['label']

# 2.4 划分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集：{X_train.shape}，验证集：{X_val.shape}，特征数：{len(feature_cols)}")

# ---------------------- 3. 随机森林参数优化（核心3个参数） ----------------------
# 参数范围
param_ranges = {
    'n_estimators': range(10, 210, 20),    # 10-200，步长20
    'max_depth': range(1, 16, 1),          # 1-15，步长1
    'min_samples_split': range(2, 22, 2)   # 2-20，步长2
}

# 存储优化结果
param_results = {}

# ---------------------- 3.1 优化1：n_estimators（决策树数量） ----------------------
print(f"\n=== 优化 n_estimators ===")
n_scores = []
for n in param_ranges['n_estimators']:
    rf = RandomForestClassifier(
        n_estimators=n, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    f1 = f1_score(y_val, rf.predict(X_val))
    n_scores.append(f1)
    print(f"n_estimators={n} → F1={f1:.4f}")

# 存储结果
param_results['n_estimators'] = {
    'params': list(param_ranges['n_estimators']),
    'scores': n_scores,
    'best_param': param_ranges['n_estimators'][np.argmax(n_scores)],
    'best_score': max(n_scores)
}

# ---------------------- 3.2 优化2：max_depth（树深度） ----------------------
print(f"\n=== 优化 max_depth ===")
best_n = param_results['n_estimators']['best_param']
depth_scores = []
for depth in param_ranges['max_depth']:
    rf = RandomForestClassifier(
        n_estimators=best_n, max_depth=depth, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    f1 = f1_score(y_val, rf.predict(X_val))
    depth_scores.append(f1)
    print(f"max_depth={depth} → F1={f1:.4f}")

param_results['max_depth'] = {
    'params': list(param_ranges['max_depth']),
    'scores': depth_scores,
    'best_param': param_ranges['max_depth'][np.argmax(depth_scores)],
    'best_score': max(depth_scores)
}

# ---------------------- 3.3 优化3：min_samples_split（分裂样本数） ----------------------
print(f"\n=== 优化 min_samples_split ===")
best_depth = param_results['max_depth']['best_param']
split_scores = []
for split in param_ranges['min_samples_split']:
    rf = RandomForestClassifier(
        n_estimators=best_n, max_depth=best_depth, min_samples_split=split,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    f1 = f1_score(y_val, rf.predict(X_val))
    split_scores.append(f1)
    print(f"min_samples_split={split} → F1={f1:.4f}")

param_results['min_samples_split'] = {
    'params': list(param_ranges['min_samples_split']),
    'scores': split_scores,
    'best_param': param_ranges['min_samples_split'][np.argmax(split_scores)],
    'best_score': max(split_scores)
}

# ---------------------- 4. 绘制优化折线图（自动适配17个事件） ----------------------
# ---------------------- 4.1 图1：n_estimators ----------------------
plt.figure(figsize=(10, 6))
plt.plot(
    param_results['n_estimators']['params'],
    param_results['n_estimators']['scores'],
    marker='o', linewidth=2, markersize=6, color='#2F5597'
)
best_n = param_results['n_estimators']['best_param']
best_score_n = param_results['n_estimators']['best_score']
plt.scatter(best_n, best_score_n, color='red', s=100, zorder=5)
plt.annotate(
    f'Best: {best_n}\nF1={best_score_n:.4f}',
    xy=(best_n, best_score_n), xytext=(best_n+10, best_score_n-0.01),
    fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
)
min_f1_n = min(param_results['n_estimators']['scores'])
max_f1_n = max(param_results['n_estimators']['scores'])
plt.ylim(min_f1_n - 0.02, max_f1_n + 0.02)
plt.xlabel('Number of Estimators (n_estimators)', fontsize=12, fontweight='bold')
plt.ylabel('Validation F1-Score', fontsize=12, fontweight='bold')
plt.title('RF Optimization: n_estimators\n(17 HPC Events)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(f"{save_path}RF_Optimization_n_estimators.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------- 4.2 图2：max_depth ----------------------
plt.figure(figsize=(10, 6))
plt.plot(
    param_results['max_depth']['params'],
    param_results['max_depth']['scores'],
    marker='s', linewidth=2, markersize=6, color='#A62E5C'
)
best_depth = param_results['max_depth']['best_param']
best_score_depth = param_results['max_depth']['best_score']
plt.scatter(best_depth, best_score_depth, color='red', s=100, zorder=5)
plt.annotate(
    f'Best: {best_depth}\nF1={best_score_depth:.4f}',
    xy=(best_depth, best_score_depth), xytext=(best_depth+0.5, best_score_depth-0.01),
    fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
)
min_f1_depth = min(param_results['max_depth']['scores'])
max_f1_depth = max(param_results['max_depth']['scores'])
plt.ylim(min_f1_depth - 0.02, max_f1_depth + 0.02)
plt.xlabel('Maximum Tree Depth (max_depth)', fontsize=12, fontweight='bold')
plt.ylabel('Validation F1-Score', fontsize=12, fontweight='bold')
plt.title('RF Optimization: max_depth\n(17 HPC Events)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(f"{save_path}RF_Optimization_max_depth.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------- 4.3 图3：min_samples_split ----------------------
plt.figure(figsize=(10, 6))
plt.plot(
    param_results['min_samples_split']['params'],
    param_results['min_samples_split']['scores'],
    marker='^', linewidth=2, markersize=6, color='#3AA17E'
)
best_split = param_results['min_samples_split']['best_param']
best_score_split = param_results['min_samples_split']['best_score']
plt.scatter(best_split, best_score_split, color='red', s=100, zorder=5)
plt.annotate(
    f'Best: {best_split}\nF1={best_score_split:.4f}',
    xy=(best_split, best_score_split), xytext=(best_split+1, best_score_split-0.01),
    fontsize=9, ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
)
min_f1_split = min(param_results['min_samples_split']['scores'])
max_f1_split = max(param_results['min_samples_split']['scores'])
plt.ylim(min_f1_split - 0.02, max_f1_split + 0.02)
plt.xlabel('Minimum Samples Split (min_samples_split)', fontsize=12, fontweight='bold')
plt.ylabel('Validation F1-Score', fontsize=12, fontweight='bold')
plt.title('RF Optimization: min_samples_split\n(17 HPC Events)', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(f"{save_path}RF_Optimization_min_samples_split.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ---------------------- 5. 输出最优参数 ----------------------
print(f"\n" + "="*60)
print("随机森林参数优化结果（17 个 HPC 事件）")
print("="*60)
print(f"1. n_estimators 最优值：{param_results['n_estimators']['best_param']} → F1={param_results['n_estimators']['best_score']:.4f}")
print(f"2. max_depth     最优值：{param_results['max_depth']['best_param']} → F1={param_results['max_depth']['best_score']:.4f}")
print(f"3. min_samples_split 最优值：{param_results['min_samples_split']['best_param']} → F1={param_results['min_samples_split']['best_score']:.4f}")
print(f"\n三张优化折线图已保存至：{save_path}")
print("="*60)