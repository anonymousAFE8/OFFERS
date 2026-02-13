import os
import json
import pandas as pd

def analyze_patterns_to_excel(folder_path, output_excel):
    # 初始化字典来存储每对T和T2设置下的统计数据
    pattern_stats = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            # 获取T和T2的值
            parts = filename.split('_')
            if len(parts) >= 6:
                T_value = float(parts[3])
                T2_value = float(parts[5].replace('.json',''))
                
                # 初始化存储统计信息的字典
                if (T_value, T2_value) not in pattern_stats:
                    pattern_stats[(T_value, T2_value)] = {"total_patterns": 0, "length_2": 0, "length_greater_than_2": 0}

                # 读取并解析JSON文件
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # 计算不同类型的pattern数量
                    for pattern in data:
                        pattern_stats[(T_value, T2_value)]["total_patterns"] += 1
                        if len(pattern) == 2:
                            pattern_stats[(T_value, T2_value)]["length_2"] += 1
                        elif len(pattern) > 2:
                            pattern_stats[(T_value, T2_value)]["length_greater_than_2"] += 1

    # 提取所有的T和T2值
    T_values = sorted(set(key[0] for key in pattern_stats.keys()))
    T2_values = sorted(set(key[1] for key in pattern_stats.keys()))

    # 创建 DataFrames 来存储统计信息
    total_patterns_df = pd.DataFrame(index=T2_values, columns=T_values)
    length_2_df = pd.DataFrame(index=T2_values, columns=T_values)
    length_greater_than_2_df = pd.DataFrame(index=T2_values, columns=T_values)

    # 填入数据到 DataFrames 中
    for (T_value, T2_value), stats in pattern_stats.items():
        total_patterns_df.at[T2_value, T_value] = stats["total_patterns"]
        length_2_df.at[T2_value, T_value] = stats["length_2"]
        length_greater_than_2_df.at[T2_value, T_value] = stats["length_greater_than_2"]

    # 将 DataFrames 导出到 Excel 不同的工作表中
    with pd.ExcelWriter(output_excel) as writer:
        total_patterns_df.to_excel(writer, sheet_name='Total Patterns')
        length_2_df.to_excel(writer, sheet_name='Length 2 Patterns')
        length_greater_than_2_df.to_excel(writer, sheet_name='Length >2 Patterns')

# 使用文件夹路径和输出文件名调用函数
folder_path = './'
output_excel = 'pattern_analysis.xlsx'
analyze_patterns_to_excel(folder_path, output_excel)