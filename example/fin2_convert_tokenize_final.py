import os
import pandas as pd
import argparse
import json
import csv

def read_sasrec_csv(file_path):
    df = pd.read_csv(file_path)
    df['sequence_item_ids'] = df['sequence_item_ids'].apply(lambda x: list(map(int, x.split(','))))
    df['sequence_ratings'] = df['sequence_ratings'].apply(lambda x: list(map(int, x.split(','))))
    df['sequence_timestamps'] = df['sequence_timestamps'].apply(lambda x: list(map(int, x.split(','))))
    return df

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def replace_patterns(df, patterns):
    for index, row in df.iterrows():
        item_ids = row['sequence_item_ids']
        ratings = row['sequence_ratings']
        timestamps = row['sequence_timestamps']
        
        i = 0
        while i < len(item_ids):
            item_id = str(item_ids[i])
            if item_id in patterns:
                pattern_info = patterns[item_id]
                
                pattern_items = list(map(int, pattern_info['pattern']))
                pattern_ratings = list(map(int, pattern_info['sequence_ratings']))
                
                if i == 0:
                    before_timestamp = 0
                else:
                    before_timestamp = timestamps[i - 1]
                    
                if i == len(item_ids) - 1:
                    after_timestamp = before_timestamp + 50 * len(pattern_items)
                else:
                    after_timestamp = timestamps[i + 1]
                    
                pattern_timestamps = [
                    int(before_timestamp + (after_timestamp - before_timestamp) * (j + 1) / (len(pattern_items) + 1))
                    for j in range(len(pattern_items))
                ]
                
                df.at[index, 'sequence_item_ids'] = item_ids[:i] + pattern_items + item_ids[i+1:]
                df.at[index, 'sequence_ratings'] = ratings[:i] + pattern_ratings + ratings[i+1:]
                df.at[index, 'sequence_timestamps'] = timestamps[:i] + pattern_timestamps + timestamps[i+1:]
                
                i += len(pattern_items) - 1
            i += 1

def convert_lists_to_strings(df):
    df['sequence_item_ids'] = df['sequence_item_ids'].apply(lambda x: ','.join(map(str, x)))
    df['sequence_ratings'] = df['sequence_ratings'].apply(lambda x: ','.join(map(str, x)))
    df['sequence_timestamps'] = df['sequence_timestamps'].apply(lambda x: ','.join(map(str, x)))
    return df

def main():
    
    parser = argparse.ArgumentParser(description='Process data with root path.')
    parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
    args = parser.parse_args()
    directory = args.root_path+'/'  # 定义所有操作所需的目录名称

    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    csv_file_name = 'sasrec_format_new.csv'
    json_file_name = 'pattern_mappings.json'
    
    # 使用 os.path.join 来构建完整路径
    csv_file_path = os.path.join(directory, csv_file_name)
    json_file_path = os.path.join(directory, json_file_name)
    
    df = read_sasrec_csv(csv_file_path)
    pattern_dict = read_json(json_file_path)
    
    replace_patterns(df, pattern_dict)
    
    # 转换列表为字符串以便输出格式正确
    df = convert_lists_to_strings(df)
    
    # 保存转换后的 DataFrame 到文件中
    output_file_name = 'sasrec_format_final_p2.csv'
    output_file_path = os.path.join(directory, output_file_name)
    df.to_csv(output_file_path, index=False)
    print(f"Data processed and saved to {output_file_path}")

    

if __name__ == "__main__":
    main()