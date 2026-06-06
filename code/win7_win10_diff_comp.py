import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

# ---------------- Win7 全部特征 ----------------
win7_feat = [
    "LLC-store_mean", "cpu-cycles_med", "cpu-cycles_mean",
    "LLC-store_med", "cache-references_mean", "L1-icache-load-misses_mean",
    "bus-cycles_med", "LLC-store-misses_med", "L1-dcache-stores_med",
    "node-stores_med", "dTLB-stores_med", "iTLB-loads_mean",
    "L1-dcache-stores_mean", "dTLB-stores_mean", "L1-dcache-load-misses_mean",
    "LLC-loads_mean", "cache-references_med", "cache-misses_med",
    "L1-icache-load-misses_med", "instructions_med", "iTLB-loads_med"
]
win7_r = [
    0.4120, 0.3845, 0.3585, 0.3505, 0.3466, 0.3369, 0.3346,
    0.3317, 0.3297, 0.3297, 0.3289, 0.3266, 0.3243, 0.3194,
    0.3163, 0.3136, 0.3130, 0.3127, 0.3065, 0.3052, 0.3009
]

# ---------------- Win10 全部特征 ----------------
win10_feat = [
    "cpu_atom_branch-load-misses__std", "cpu_atom_branch-load-misses__max",
    "cpu_atom_branch-loads__std", "cpu_atom_branch-loads__max",
    "cpu_atom_branch-misses__std", "cpu_atom_branch-misses__max",
    "cpu_atom_bus-cycles__std", "cpu_atom_bus-cycles__max",
    "cpu_atom_cache-references__std", "cpu_atom_cache-references__max",
    "cpu_atom_cpu-cycles__std", "cpu_atom_cpu-cycles__max",
    "cpu_atom_dTLB-load-misses__std", "cpu_atom_dTLB-load-misses__max",
    "cpu_atom_iTLB-load-misses__std", "cpu_atom_iTLB-load-misses__max",
    "cpu_atom_LLC-loads__max",
    "cpu_core_branch-load-misses__std", "cpu_core_branch-load-misses__max",
    "cpu_core_branch-loads__std", "cpu_core_branch-misses__std",
    "cpu_core_branch-misses__max", "cpu_core_bus-cycles__std",
    "cpu_core_bus-cycles__max", "cpu_core_cache-misses__std",
    "cpu_core_cache-misses__max", "cpu_core_cache-references__std",
    "cpu_core_cache-references__max", "cpu_core_cpu-cycles__std",
    "cpu_core_cpu-cycles__max", "cpu_core_cpu-cycles__med",
    "cpu_core_dTLB-load-misses__std", "cpu_core_iTLB-load-misses__med",
    "cpu_core_L1-dcache-load-misses__std", "cpu_core_L1-icache-load-misses__med",
    "cpu_core_LLC-loads__std"
]
win10_r = [
    0.8422, 0.8303, 0.5130, 0.5930, 0.5181, 0.5602, 0.6931, 0.6994,
    0.6415, 0.7101, 0.6866, 0.7051, 0.5155, 0.5309, 0.5063, 0.5603,
    0.4793, 0.8248, 0.8213, 0.4493, 0.5398, 0.5363, 0.6431, 0.6494,
    0.6963, 0.6290, 0.6784, 0.6865, 0.6235, 0.6447, 0.3355, 0.5544,
    0.3031, 0.5977, 0.3065, 0.4566
]

# ---------------- 绘图 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 10))

# Win7
ax1.barh(win7_feat[::-1], win7_r[::-1], color='#3366cc', height=0.6)
ax1.set_xlabel('Absolute Correlation (abs_r)', weight='bold', fontsize=9)
ax1.set_title('Windows 7 (i7-7500U)\nAll Significant HPC Features', weight='bold', fontsize=10)
ax1.set_xlim(0, 0.5)
ax1.grid(axis='x', linestyle=':', alpha=0.4)
# 关键：缩小y轴标签字体
ax1.tick_params(axis='y', labelsize=7)

# Win10
ax2.barh(win10_feat[::-1], win10_r[::-1], color='#cc3333', height=0.6)
ax2.set_xlabel('Absolute Correlation (abs_r)', weight='bold', fontsize=9)
ax2.set_title('Windows 10 (Ultra9 185H)\nAll Significant HPC Features', weight='bold', fontsize=10)
ax2.set_xlim(0, 0.9)
ax2.grid(axis='x', linestyle=':', alpha=0.4)
# 关键：缩小y轴标签字体
ax2.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig('cross_platform_all_hpc_features.png', dpi=300, bbox_inches='tight')
plt.savefig('cross_platform_all_hpc_features.pdf', bbox_inches='tight')
plt.show()