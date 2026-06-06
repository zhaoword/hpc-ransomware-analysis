import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# 路径配置
# ==============================================================================
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\cluster_win7_win10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入文件路径（特征矩阵和统计文件）
BASE_PATH = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\feature_metrix"

# Win7 初筛结果（9个特征）
WIN7_FILTERED_FEATURES = [
    "node-loads_std",
    "cpu-cycles_med",
    "LLC-store-misses_mean",
    "LLC-store_mean",
    "LLC-store-misses_std",
    "node-stores_mean",
    "cache-misses_std",
    "LLC-store_std",
    "node-stores_std"
]

# Win10 初筛结果（14个特征）
WIN10_FILTERED_FEATURES = [
    "cpu_core_cpu-cycles__max",
    "cpu_core_bus-cycles__max",
    "cpu_core_cache-references__std",
    "cpu_core_cache-references__max",
    "cpu_atom_cpu-cycles__std",
    "cpu_atom_bus-cycles__std",
    "cpu_core_cache-misses__std",
    "cpu_atom_bus-cycles__max",
    "cpu_atom_cpu-cycles__max",
    "cpu_atom_cache-references__max",
    "cpu_core_branch-load-misses__max",
    "cpu_core_branch-load-misses__std",
    "cpu_atom_branch-load-misses__max",
    "cpu_atom_branch-load-misses__std"
]


# ================== 随机森林置换重要性 ==================
def get_permutation_importance(feat_matrix_path, filtered_features):
    """计算置换重要性"""
    df_feat = pd.read_csv(feat_matrix_path)
    available_feats = [f for f in filtered_features if f in df_feat.columns]
    print(f"  有效特征数：{len(available_feats)}")

    X = df_feat[available_feats]
    y = df_feat["label"]

    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X, y)

    # 置换重要性
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, scoring='accuracy')

    df_rf = pd.DataFrame({
        "feature": X.columns,
        "importance": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    })
    df_rf = df_rf.sort_values("importance", ascending=False)

    return df_rf


# ================== 肘部图函数 ==================
def plot_elbow_curve(X_cluster, platform, output_dir):
    """绘制肘部图确定最优 K 值"""
    n_features = X_cluster.shape[0]  # 特征数量
    max_k = min(10, n_features)  # K 最大值不超过特征数
    K_range = range(1, max_k + 1)

    sse = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.title(f'Elbow Method for Optimal K - {platform.upper()}', fontsize=14)

    # 如果 K=4 在范围内，标记
    if 4 in K_range:
        plt.axvline(x=4, color='r', linestyle='--', label='K=4 (selected)')
        plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"elbow_curve_{platform}.png"), dpi=300)
    plt.close()
    print(f"✅ 肘部图已保存: elbow_curve_{platform}.png (K range: 1-{max_k})")


# ================== 聚类分析 ==================
def run_clustering(platform, df_importance, feature_matrix_path, filtered_features):
    print(f"\n{'=' * 50}")
    print(f"📊 处理平台：{platform.upper()}")
    print(f"{'=' * 50}")

    # 获取有效特征
    df_feat = pd.read_csv(feature_matrix_path)
    available_feats = [f for f in filtered_features if f in df_feat.columns]
    df_importance = df_importance[df_importance["feature"].isin(available_feats)]

    features = df_importance["feature"].values

    # 读取统计信息（获取 abs_r）
    stat_path = f"{BASE_PATH}/HPC_25events_FINAL_ANALYSIS.csv" if platform == "win7" else f"{BASE_PATH}/HPC_42events_FINAL_with_FDR.csv"
    df_stat = pd.read_csv(stat_path)
    df_stat = df_stat[["feature", "abs_r"]]
    df_importance = df_importance.merge(df_stat, on="feature", how="left")

    # 计算特征间相关系数矩阵
    df_sub = df_feat[features]
    corr_matrix = df_sub.corr().abs()

    # 构建聚类输入：重要性 + 与其他特征的相关性
    imp_dict = dict(zip(df_importance["feature"], df_importance["importance"]))
    X_cluster = []
    for f in features:
        imp = imp_dict[f]
        sim = corr_matrix[f].values
        X_cluster.append([imp] + sim.tolist())

    X_cluster = np.array(X_cluster)
    X_cluster = StandardScaler().fit_transform(X_cluster)

    # ================== 绘制肘部图 ==================
    plot_elbow_curve(X_cluster, platform, OUTPUT_DIR)

    # K-means 聚类（K=4）
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster)

    # 记录聚类结果
    df_cluster = pd.DataFrame({
        "feature": features,
        "cluster": labels + 1
    })

    # 合并完整信息
    df_merge = df_importance.merge(df_cluster, on="feature")

    # 计算综合得分（重要性 × 0.6 + 相关性 × 0.4）
    df_merge["composite_score"] = df_merge["importance"] * 0.6 + df_merge["abs_r"] * 0.4

    # 打印每个特征所属簇
    print(f"\n📌 特征聚类分组 (Cluster 1~{n_clusters}):")
    for c in range(1, n_clusters + 1):
        print(f"\n  Cluster {c}:")
        feats_in_cluster = df_merge[df_merge["cluster"] == c]["feature"].tolist()
        for f in feats_in_cluster:
            imp_val = df_merge[df_merge["feature"] == f]["importance"].values[0]
            print(f"    - {f} (imp={imp_val:.5f})")

    # 每个簇选综合得分最高的特征
    selected = []
    for c in range(1, n_clusters + 1):
        best = df_merge[df_merge["cluster"] == c].sort_values("composite_score", ascending=False).iloc[0]
        selected.append(best)
    df_selected = pd.DataFrame(selected)

    print(f"\n✅ 每簇选出的代表性特征:")
    for _, row in df_selected.iterrows():
        print(f"   Cluster {row['cluster']}: {row['feature']} (score={row['composite_score']:.4f})")

    # 验证选中特征间的最大相关性
    sel_feats = df_selected["feature"].tolist()
    final_corr = df_feat[sel_feats].corr().abs()
    max_corr = final_corr.values[np.triu_indices_from(final_corr.values, k=1)].max() if len(sel_feats) > 1 else 0
    print(f"\n✅ 选中特征间最大相关系数: {max_corr:.4f}")

    # 保存结果
    df_merge.to_csv(os.path.join(OUTPUT_DIR, f"CLUSTER_FULL_{platform}.csv"), index=False, encoding="utf-8-sig")
    df_selected.to_csv(os.path.join(OUTPUT_DIR, f"CLUSTER_TOP4_{platform}.csv"), index=False, encoding="utf-8-sig")

    # ================== 绘图 ==================
    # 图1：聚类散点图（PCA降维）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)

    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(n_clusters):
        mask = df_merge["cluster"] == i + 1
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=100, c=colors[i], label=f'Cluster {i + 1}', alpha=0.7)
        for idx, (x, y) in enumerate(zip(X_pca[mask, 0], X_pca[mask, 1])):
            feat_name = df_merge[mask]["feature"].iloc[idx]
            if len(feat_name) > 30:
                feat_name = feat_name[:27] + "..."
            plt.annotate(feat_name, (x, y), fontsize=8, alpha=0.8)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{platform.upper()} Feature Clustering (K=4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"clustering_scatter_{platform}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：最终选中特征的综合得分
    plt.figure(figsize=(10, 6))
    df_plot = df_selected.sort_values("composite_score", ascending=True)
    plt.barh(df_plot["feature"], df_plot["composite_score"], color='#4477bb')
    plt.xlabel("Composite Score (0.6×Importance + 0.4×|r|)")
    plt.title(f"{platform.upper()} Selected Features per Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_final_features_{platform}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return df_merge, df_selected, max_corr


# ================== 主程序 ==================
print("=" * 60)
print("聚类分析（基于置换重要性 + 初筛特征）")
print(f"输出目录：{OUTPUT_DIR}")
print("=" * 60)

# 计算置换重要性
print("\n📊 计算 Win7 置换重要性...")
df_imp7 = get_permutation_importance(
    f"{BASE_PATH}/HPC_sample_feature_matrix_win7.csv",
    WIN7_FILTERED_FEATURES
)

print("\n📊 计算 Win10 置换重要性...")
df_imp10 = get_permutation_importance(
    f"{BASE_PATH}/HPC_sample_feature_matrix-win10.csv",
    WIN10_FILTERED_FEATURES
)

# 运行聚类
df_full7, df_sel7, max_r7 = run_clustering(
    "win7", df_imp7,
    f"{BASE_PATH}/HPC_sample_feature_matrix_win7.csv",
    WIN7_FILTERED_FEATURES
)

df_full10, df_sel10, max_r10 = run_clustering(
    "win10", df_imp10,
    f"{BASE_PATH}/HPC_sample_feature_matrix-win10.csv",
    WIN10_FILTERED_FEATURES
)

print("\n" + "=" * 60)
print("🎉 聚类分析完成！")
print("=" * 60)
print(f"📁 输出目录：{OUTPUT_DIR}")
print(f"📄 生成的文件：")
print(f"   - elbow_curve_win7.png / win10.png（肘部图）")
print(f"   - CLUSTER_FULL_win7.csv / win10.csv")
print(f"   - CLUSTER_TOP4_win7.csv / win10.csv")
print(f"   - clustering_scatter_win7.png / win10.png")
print(f"   - cluster_final_features_win7.png / win10.png")