import pandas as pd
import numpy as np
import os

# ==================== 路径配置 ====================
# Win7 事件文件夹（统一后的数据）
WIN7_EVENTS_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\merged_folder_common_win7"
# Win10 事件文件夹
WIN10_EVENTS_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\merged_result_win10"
# 输出目录（与您的随机森林脚本路径一致）
OUTPUT_DIR = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\feature_metrix"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENCODING = "latin-1"


# ==================== 构建特征矩阵 ====================
def build_feature_matrix(events_dir, platform_name):
    print(f"\n处理 {platform_name}...")

    # 获取所有 CSV 文件
    files = [f for f in os.listdir(events_dir) if f.endswith('.csv')]
    print(f"  找到 {len(files)} 个事件文件")

    feature_dict = {}
    label = None

    for fname in files:
        path = os.path.join(events_dir, fname)
        df = pd.read_csv(path, encoding=ENCODING)

        if 'label' not in df.columns:
            continue

        if label is None:
            label = df['label'].values

        data_cols = [c for c in df.columns if c.startswith('data_')]
        X = df[data_cols].values

        event_name = os.path.splitext(fname)[0]
        feature_dict[f"{event_name}_mean"] = np.nanmean(X, axis=1)
        feature_dict[f"{event_name}_std"] = np.nanstd(X, axis=1, ddof=1)
        feature_dict[f"{event_name}_max"] = np.nanmax(X, axis=1)
        feature_dict[f"{event_name}_med"] = np.nanmedian(X, axis=1)

    df_feature = pd.DataFrame(feature_dict)
    df_feature['label'] = label

    print(f"  特征矩阵形状: {df_feature.shape}")
    print(f"  样本数: {len(df_feature)}, 特征数: {len(df_feature.columns) - 1}")

    return df_feature


# ==================== 执行 ====================
print("=" * 60)
print("构建样本特征矩阵")
print("=" * 60)

win7_features = build_feature_matrix(WIN7_EVENTS_DIR, "Win7")
win10_features = build_feature_matrix(WIN10_EVENTS_DIR, "Win10")

# ==================== 保存 ====================
win7_output = os.path.join(OUTPUT_DIR, "HPC_sample_feature_matrix_win7.csv")
win10_output = os.path.join(OUTPUT_DIR, "HPC_sample_feature_matrix-win10.csv")

win7_features.to_csv(win7_output, index=False)
win10_features.to_csv(win10_output, index=False)

print(f"\n✅ 已保存: {win7_output}")
print(f"✅ 已保存: {win10_output}")

# 验证
print("\n验证 Win7 文件前几行:")
print(win7_features.iloc[:3, :5])
print("\nlabel 分布:")
print(win7_features['label'].value_counts())