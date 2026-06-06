import re
import os
import csv

# ===================== 一键切换 =====================
SELECT_MS = 100
# ====================================================

base_folder = r'D:\data_doctor\win10_doctor\win7_win10_join\yz_three_family\win7_1'
output_file = os.path.join(base_folder, f'final_rs_{SELECT_MS}ms_FINAL_win7.csv')

# 5 个事件（已补全逗号）
OUTPUT_ORDER = [
    'node-loads',
    'cpu-cycles',
    'node-store',
    'cache-misses',
    'instructions',
]

# ======================================================================
# 事件识别：5个事件全匹配
# ======================================================================
def get_event(line):
    line = line.lower().strip()
    if 'node-loads' in line:
        return 'node-loads'
    if 'cpu-cycles' in line:
        return 'cpu-cycles'
    if 'node-store' in line:
        return 'node-store'
    if 'cache-misses' in line:
        return 'cache-misses'
    if 'instructions' in line:
        return 'instructions'
    return 'unknown'

# ======================================================================
# 5事件 完整处理
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

        # 自动清洗：跳过脏表头
        cleaned = []
        for line in raw_lines:
            s = line.strip()
            if not s:
                continue
            if '#' in s and not any(c.isdigit() for c in s):
                continue
            cleaned.append(s)

        # 5个事件容器
        event_data = {
            'node-loads': [],
            'cpu-cycles': [],
            'node-store': [],
            'cache-misses': [],
            'instructions': []
        }

        # 5事件轮询：0→1→2→3→4→0...
        for i, row in enumerate(cleaned):
            row = row.replace(',', '').strip()
            parts = re.split(r'\s+', row)
            if len(parts) < 2:
                continue

            val = parts[1]
            mod = i % 5  # 👈 改成 5

            if mod == 0:
                event_data['node-loads'].append(val)
            elif mod == 1:
                event_data['cpu-cycles'].append(val)
            elif mod == 2:
                event_data['node-store'].append(val)
            elif mod == 3:
                event_data['cache-misses'].append(val)
            else:
                event_data['instructions'].append(val)

        # 统计
        current_max = max(len(v) for v in event_data.values())
        if current_max > max_data_length:
            max_data_length = current_max

        sample_data_list.append({
            'sample_id': filename[:-4],
            'label': 0,
            'event_data': event_data
        })

        # 打印5个事件长度
        print(f"✅ {filename} | "
              f"node-loads:{len(event_data['node-loads'])} | "
              f"cpu-cycles:{len(event_data['cpu-cycles'])} | "
              f"node-store:{len(event_data['node-store'])} | "
              f"cache-misses:{len(event_data['cache-misses'])} | "
              f"instructions:{len(event_data['instructions'])}")

    except Exception as e:
        print(f"❌ {filename} 错误：{str(e)}")

# ======================================================================
# 输出 5事件 CSV
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

print("\n🎉 5个事件处理完成！")