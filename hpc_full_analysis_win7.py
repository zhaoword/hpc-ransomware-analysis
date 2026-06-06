import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection   # [NEW] 导入 FDR
import warnings
warnings.filterwarnings('ignore')

# ====================== Paper Font Configuration ======================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ====================== Path Configuration ======================
CSV_FOLDER = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\merged_folder_common_win7"
SAVE_DIR   = r"D:\data_doctor\win10_doctor\win7_win10_join\new_10s_data_win7_win10\win7"
ENCODING   = "latin-1"
os.makedirs(SAVE_DIR, exist_ok=True)
# ====================================================================

def load_all_events(folder):
    """Load all HPC event CSV files from target directory"""
    events = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(folder, fname)
        try:
            df = pd.read_csv(path, encoding=ENCODING)
        except:
            continue
        if "label" not in df.columns:
            continue
        data_cols = [c for c in df.columns if c.startswith("data_")]
        if not data_cols:
            continue
        event_name = os.path.splitext(fname)[0]
        events[event_name] = df
    return events

def compute_agg_feature(df):
    """Calculate aggregated statistical features: mean, std, max, median"""
    data_cols = [c for c in df.columns if c.startswith("data_")]
    x = df[data_cols].values
    return {
        "mean": np.mean(x, axis=1),
        "std": np.std(x, axis=1, ddof=1),
        "max": np.max(x, axis=1),
        "med": np.median(x, axis=1)
    }

def cohen_d(x0, x1):
    """Compute Cohen's d effect size between benign and malicious groups"""
    n0, n1 = len(x0), len(x1)
    s0, s1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
    pooled = np.sqrt(((n0-1)*s0 + (n1-1)*s1) / (n0+n1-2)) if (n0+n1-2)>0 else 1e-9
    return (np.mean(x1)-np.mean(x0))/pooled if pooled>1e-9 else 0.0

# ====================== Correlation & Significance Level ======================
def level_correlation(abs_r):
    """Classify correlation strength by absolute value"""
    if abs_r >= 0.6:
        return "Strong"
    elif abs_r >= 0.4:
        return "Moderate"
    else:
        return "Weak"

def level_significance(p):
    """Classify statistical significance"""
    if p < 1e-10:
        return "Highly Significant"
    elif p < 0.05:
        return "Significant"
    else:
        return "Not Significant"

def level_fold_change(fc):
    """Classify fold change difference level"""
    if fc >= 2.0:
        return "High"
    elif fc >= 1.5:
        return "Moderate"
    else:
        return "Low"

# [NEW] Cohen's d effect size classification
def level_cohen_d(d):
    abs_d = abs(d)
    if abs_d >= 0.8:
        return "Large"
    elif abs_d >= 0.5:
        return "Medium"
    else:
        return "Small"

# ===============================================================================

def analyze_all(events):
    """Comprehensive analysis: correlation, significance, fold change, effect size"""
    rows = []
    feat_vectors = {}  # Store feature vectors for correlation matrix construction

    for evt, df in events.items():
        feat = compute_agg_feature(df)
        y = df["label"].values
        for ftype, vec in feat.items():
            mask = np.isfinite(vec)
            vec, y_f = vec[mask], y[mask]
            if len(vec) < 20:
                continue

            vec0 = vec[y == 0]
            vec1 = vec[y == 1]
            mu0, mu1 = np.mean(vec0), np.mean(vec1)

            if mu0 < 1e-9:
                fc, diff, gp = np.nan, np.nan, np.nan
            else:
                fc = mu1 / mu0
                diff = mu1 - mu0
                gp = (diff / mu0) * 100

            # Use raw real correlation value (keep positive and negative)
            r, p = pointbiserialr(y_f, vec)
            abs_r = abs(r)
            d = cohen_d(vec0, vec1)

            feat_name = f"{evt}_{ftype}"
            feat_vectors[feat_name] = vec

            rows.append({
                "event": evt,
                "stat": ftype,
                "feature": feat_name,
                "corr_r": round(r, 4),          # Real correlation with positive/negative
                "abs_r": round(abs_r, 4),
                "corr_level": level_correlation(abs_r),
                "p_value": round(p, 6),
                "sig_level": level_significance(p),
                "fold_change": round(fc, 4) if not np.isnan(fc) else 0,
                "fc_level": level_fold_change(fc) if not np.isnan(fc) else "Low",
                "growth_pct": round(gp, 4) if not np.isnan(gp) else 0,
                "mean_benign": round(mu0, 4),
                "mean_malicious": round(mu1, 4),
                "cohen_d": round(d, 4),
                "cohen_d_level": level_cohen_d(d)   # [NEW] Cohen's d classification
            })

    df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)

    # [NEW] FDR multiple testing correction (Benjamini-Hochberg)
    reject, pvals_corrected = fdrcorrection(df['p_value'].values, alpha=0.05)
    df['p_fdr'] = pvals_corrected
    df['sig_fdr'] = ['Significant' if r else 'Not Significant' for r in reject]

    # Build feature pairwise correlation matrix
    feat_names = list(feat_vectors.keys())
    valid_vecs = [feat_vectors[fn] for fn in feat_names]
    feature_matrix = np.vstack(valid_vecs)
    corr_mat = pd.DataFrame(np.corrcoef(feature_matrix), index=feat_names, columns=feat_names)
    return df, corr_mat

def analyze_redundancy(corr_mat, save_dir):
    """Feature redundancy analysis based on pairwise correlation coefficient"""
    corr_diag = corr_mat.copy()
    np.fill_diagonal(corr_diag.values, np.nan)
    mean_corr = corr_diag.mean().mean()
    high_redundant_pairs = []

    for i in range(len(corr_diag)):
        for j in range(i+1, len(corr_diag)):
            val = corr_diag.iloc[i, j]
            if abs(val) >= 0.7:
                high_redundant_pairs.append((corr_diag.index[i], corr_diag.columns[j], round(val, 3)))

    high_redundant_pairs = sorted(high_redundant_pairs, key=lambda x: abs(x[2]), reverse=True)
    low_redundant = corr_diag.mean().sort_values()[:5].index.tolist()
    corr_diag.to_csv(os.path.join(save_dir, "redundancy_correlation_matrix.csv"), encoding="utf-8-sig")

    print("\n" + "="*50)
    print("Feature Redundancy Analysis Result")
    print("="*50)
    print(f"Average Inter-Feature Correlation: {mean_corr:.3f}")
    print(f"Rule: 0~0.3 Low | 0.3~0.7 Moderate | ≥0.7 High\n")
    print(f"High Redundancy Feature Pairs (|r| ≥ 0.7):")
    for a, b, v in high_redundant_pairs[:10]:
        print(f"   {a} ↔ {b}  |  r = {v}")
    print(f"\nTop 5 Low Redundancy Features:")
    for idx, e in enumerate(low_redundant, 1):
        print(f"   {idx}. {e}")
    print("="*50)

def plot_final_paper_figures(df, corr_mat, save_dir):
    """Generate publication-ready figures with real correlation values"""
    os.makedirs(save_dir, exist_ok=True)
    df_best = df.loc[df.groupby("event")["abs_r"].idxmax()].copy()

    # 1. Correlation Coefficient vs Fold Change Scatter Plot
    plt.figure(figsize=(11,7))
    plt.scatter(df_best["corr_r"], df_best["fold_change"], s=110, color="#3498db", alpha=0.8, edgecolors="white")
    for _, row in df_best.iterrows():
        plt.annotate(row["event"], (row["corr_r"], row["fold_change"]), fontsize=9)
    plt.axvline(x=0, color="black", linewidth=1.2)
    plt.axvline(x=0.6, color="red", linestyle="--", label="Strong Positive (r ≥ 0.6)")
    plt.axvline(x=-0.6, color="red", linestyle="--", label="Strong Negative (r ≤ -0.6)")
    plt.axhline(y=2.0, color="green", linestyle="--", label="High Fold Change (≥ 2.0)")
    plt.xlabel("Correlation Coefficient (r)", fontsize=12)
    plt.ylabel("Fold Change", fontsize=12)
    plt.title("HPC Events: Correlation vs Fold Change", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_vs_fold.png"), dpi=300)
    plt.close()

    # 2. Top 10 Correlation Bar Plot (Real Positive/Negative Value)
    top10 = df_best.reindex(df_best["abs_r"].abs().nlargest(10).index)
    plt.figure(figsize=(12,6))
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top10["corr_r"]]
    plt.bar(top10["event"], top10["corr_r"], color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Correlation Coefficient (r)", fontsize=12)
    plt.title("Top 10 HPC Events with the Strongest Correlation to Ransomware Labels(win7)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top10_correlation_real.png"), dpi=300)
    plt.close()

    # 3. Top 10 Fold Change Bar Plot
    fc10 = df_best.nlargest(10, "fold_change")
    plt.figure(figsize=(10,5))
    plt.bar(fc10["event"], fc10["fold_change"], color="#2ecc71")
    plt.xticks(rotation=30, ha="right")
    plt.title("Top 10 HPC Events by Fold Change", fontsize=13)
    plt.ylabel("Fold Change", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top10_fold_change.png"), dpi=300)
    plt.close()

    # 4. Feature Redundancy Heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_mat, cmap="RdBu_r", center=0, linewidths=0.3, annot=False)
    plt.title("HPC Feature Redundancy Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_redundancy_heatmap.png"), dpi=300)
    plt.close()

def build_sample_feature_matrix(events, save_dir):
    """Construct sample-level feature matrix (one row corresponds to one sample)"""
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
    out_path = os.path.join(save_dir, "HPC_sample_feature_matrix.csv")
    df_feature.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Sample feature matrix saved successfully.")

# [NEW] Export event list for appendix
def export_event_list(df, save_dir):
    events = sorted(df['event'].unique())
    pd.Series(events).to_csv(os.path.join(save_dir, "HPC_event_list.csv"), index=False, header=False, encoding="utf-8-sig")
    print(f"Event list exported: {len(events)} events.")

# ====================== Main Program Entry ======================
if __name__ == "__main__":
    events = load_all_events(CSV_FOLDER)
    res_df, corr_mat = analyze_all(events)

    res_df.to_csv(os.path.join(SAVE_DIR, "HPC_25events_FINAL_ANALYSIS.csv"),
                  index=False, encoding="utf-8-sig")

    build_sample_feature_matrix(events, SAVE_DIR)
    analyze_redundancy(corr_mat, SAVE_DIR)
    plot_final_paper_figures(res_df, corr_mat, SAVE_DIR)
    export_event_list(res_df, SAVE_DIR)   # [NEW] 输出事件列表

    print("\n===== Top 10 Selected HPC Features =====")
    best10 = res_df.head(10)[["feature","corr_r","fold_change","corr_level","sig_level","cohen_d","cohen_d_level"]]
    print(best10.to_string(index=False))

    # [NEW] Print summary of Cohen's d effect sizes
    print("\n===== Cohen's d Effect Size Summary (Win7) =====")
    d_counts = res_df['cohen_d_level'].value_counts()
    print(d_counts)

    print("\nAll analysis tasks completed!")