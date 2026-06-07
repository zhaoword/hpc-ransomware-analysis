# hpc-ransomware-analysis

Code and data for HPC-based ransomware behavior analysis on Windows 7 (Intel i7-7500U) and Windows 10 (Intel Core Ultra 9 185H).

**Paper:** Hardware Performance Counter Analysis of Ransomware Behavior: Observed Inverse Correlations Across Heterogeneous x86 Platforms (Applied Sciences, 2026)
---
## Repository Structure
hpc-ransomware-analysis/
├── README.md
├── LICENSE
├── code/ # Analysis scripts
│ ├── hpc_full_analysis_win7.py # Win7 correlation analysis (Cohen's d, FDR)
│ ├── hpc_full_anlysisi_win10.py # Win10 correlation analysis
│ ├── redundance_win7_win10.py # Redundancy analysis & clustering
│ ├── phase_threshold.py # Stage-specific threshold calculation
│ ├── phase_features_class.py # Phase-level feature classification
│ ├── offline_traceback_all.py # Offline retrospective validation
│ ├── forrest_score_11.py # Random Forest with permutation importance
│ ├── feature_first.py # Initial feature screening
│ └── feature_metric.py # Feature extraction from raw data
├── data/ # Data files
│ ├── sample/ # Sample raw event traces (5 each platform)
│ ├── HPC_25events_FINAL_ANALYSIS.csv # Win7 correlation results
│ ├── HPC_42events_FINAL_with_FDR.csv # Win10 correlation results (with FDR)
│ ├── redundancy_correlation_matrix_win7.csv # Win7 feature correlation matrix
│ ├── HPC_feature_correlation_matrix_win10.csv # Win10 feature correlation matrix
│ ├── final_rs_unseen_100ms_FINAL_win7.csv # Win7 leave-one-family-out data
│ ├── final_rs_unseen_100ms_FINAL_win10.csv # Win10 leave-one-family-out data
│ ├── win7_hardware_event.csv # Win7 feature matrix
│ ├── win10_hardware_event.csv # Win10 feature matrix
│ └── 裸机试验.xls # Bare-metal validation data
## Dependencies
Python 3.8+、pandas, numpy, scipy、scikit-learn、matplotlib, seaborn、statsmodels、openpyxl (for Excel files)

## Data Description
          File	                      Description
win7_hardware_event.csv	Win7          feature matrix (25 events × 4 statistics = 100 features)
win10_hardware_event.csv	Win10       feature matrix (42 metrics × 4 statistics = 168 features)
HPC_25events_FINAL_ANALYSIS.csv	      Win7 correlation results (Pearson r, Cohen's d, fold change)
HPC_42events_FINAL_with_FDR.csv	      Win10 correlation results with FDR correction
final_rs_unseen_*.csv	                Leave-one-family-out validation data (51 unseen samples)
裸机试验.xls	                        Bare-metal vs. VM validation data

## Data Availability
The raw ransomware samples cannot be publicly shared due to security and legal restrictions. The pre-processed HPC feature matrices and all analysis code are available in this repository.

## Citation
If you use this code or data, please cite:
Zhao, E., Zhu, Z. Hardware Performance Counter Analysis of Ransomware Behavior: Observed Inverse Correlations Across Heterogeneous x86 Platforms. Applied Sciences, 2026.
