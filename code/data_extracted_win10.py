import re
import os
import csv

# ===================== 一键切换 =====================
SELECT_MS = 100
# ====================================================

# Win10 路径（你可以自己改）
base_folder = r'D:\data_doctor\win10_doctor\win7_win10_join\yz_three_family\rs_100ms_10s_3_win10'
output_file = os.path.join(base_folder, f'final_rs_{SELECT_MS}ms_FINAL_win10.csv')

# 你提供的 10 个事件（完全对应你的 EVENT_CONFIG）
EVENT_CONFIG = [
    ("cpu-cycles_atom", "cpu_atom/cpu-cycles"),
    ("cpu-cycles_core", "cpu_core/cpu-cycles"),
    ("bus-cycles_atom", "cpu_atom/bus-cycles"),
    ("bus-cycles_core", "cpu_core/bus-cycles"),
    ("cache-references_atom", "cpu_atom/cache-references"),
    ("cache-references_core", "cpu_core/cache-references"),
    ("cache_misses_atom", "cpu_atom/cache-misses"),
    ("cache_misses_core", "cpu_core/cache-misses"),
    ("branch-load-misses_atom", "cpu_atom/branch-load-misses"),
    ("branch-load-misses_core", "cpu_core/branch-load-misses"),
]

# 输出顺序（直接用上面 10 个）
OUTPUT_ORDER = [name for name, _ in EVENT_CONFIG]

# ======================================================================
# 事件识别：自动匹配 10 个事件
# ======================================================================
def get_event(line):
    line = line.lower().strip()
    for event_name, match_str in EVENT_CONFIG:
        if match_str.lower() in line:
            return event_name
    return "unknown"

# ======================================================================
# 10 事件 完整处理
# ======================================================================
max_data_length = 0
sample_data_list = []

for filename in os.listdir(base_folder):
    if not filename.endswith('.csv'):
        continue

    file_path = os.path.join(base_folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        # 清洗：跳过脏表头、空行
        cleaned = []
        for line in raw_lines:
            s = line.strip()
            if not s:
                continue
            if '#' in s and not any(c.isdigit() for c in s):
                continue
            cleaned.append(s)

        # 初始化 10 个事件容器
        event_data = {name: [] for name, _ in EVENT_CONFIG}

        # ==============================
        # 10 事件轮询：0~9 循环
        # ==============================
        for i, row in enumerate(cleaned):
            row = row.replace(',', '').strip()
            parts = re.split(r'\s+', row)
            if len(parts) < 2:
                continue

            val = parts[1]
            mod = i % 10  # 10 个事件 取模 10

            # 自动对应 10 个事件位置
            target_name = OUTPUT_ORDER[mod]
            event_data[target_name].append(val)

        # 统计最大长度
        current_max = max(len(v) for v in event_data.values())
        if current_max > max_data_length:
            max_data_length = current_max

        # 保存样本
        sample_data_list.append({
            'sample_id': filename[:-4],
            'label': 0,
            'event_data': event_data
        })

        # 打印日志
        print(f"✅ {filename} | 10事件均已提取")

    except Exception as e:
        print(f"❌ {filename} 错误：{str(e)}")

# ======================================================================
# 输出最终 CSV
# ======================================================================
header = ['sample_id', 'label']
for e in OUTPUT_ORDER:
    for t in range(1, max_data_length + 1):
        header.append(f'{e}_t{t}')

all_samples = []
for sample in sample_data_list:
    row = [sample['sample_id'], sample['label']]
    for e in OUTPUT_ORDER:
        row += sample['event_data'][e][:max_data_length]
    all_samples.append(row)

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_samples)

print("\n🎉 WIN10 —— 10 个事件全部提取完成！")