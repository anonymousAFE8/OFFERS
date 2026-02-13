import pandas as pd
import json
import argparse

def read_sasrec_format_csv(file_path):
    df = pd.read_csv(file_path)
    
    # 解析所有用户ID和物品ID，并去除重复
    unique_user_ids = df['user_id'].unique()
    unique_item_ids = set()

    for sequence in df['sequence_item_ids']:
        unique_item_ids.update(map(int, sequence.split(',')))

    # 为用户和物品ID生成排序并创建映射字典
    sorted_user_ids = sorted(unique_user_ids)
    user_id_mapping = {int(orig_id): new_id + 1 for new_id, orig_id in enumerate(sorted_user_ids)}

    sorted_item_ids = sorted(unique_item_ids)
    item_id_mapping = {int(orig_id): new_id + 1 for new_id, orig_id in enumerate(sorted_item_ids)}

    # 用于存储重映射后的数据
    data = []
    for _, row in df.iterrows():
        original_user_id = int(row['user_id'])
        new_user_id = user_id_mapping[original_user_id]

        original_item_ids = map(int, row['sequence_item_ids'].split(','))
        for item_id in original_item_ids:
            new_item_id = item_id_mapping[item_id]
            data.append(f"{new_user_id} {new_item_id}")

    return data, user_id_mapping, item_id_mapping

def save_to_txt(data, output_file_path):
    with open(output_file_path, 'w') as f:
        for line in data:
            f.write(line + '\n')

def save_id_mapping(mapping, mapping_file_path):
    with open(mapping_file_path, 'w') as f:
        json.dump(mapping, f, indent=4)

# 示例使用
parser = argparse.ArgumentParser(description='Process data with root path.')
parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
parser.add_argument('--dataset', type=str, required=True, help='The root directory path for reading and writing files.')
parser.add_argument('--target_path', type=str, required=True, help='SASRec.pytorch-main')
args = parser.parse_args()


csv_file_path = args.root_path+'/sasrec_format.csv'  # 请替换为您的实际文件路径
txt_file_path = args.target_path+'/python/data/'+args.dataset+'.txt'  # 输出数据文件路径
user_mapping_file_path = args.target_path+'/python/data/'+args.dataset+'_user_id_mapping.json'  # 用户ID映射字典文件路径
item_mapping_file_path = args.target_path+'/python/data/'+args.dataset+'_item_id_mapping.json'  # 物品ID映射字典文件路径

data, user_id_mapping, item_id_mapping = read_sasrec_format_csv(csv_file_path)
save_to_txt(data, txt_file_path)
save_id_mapping(user_id_mapping, user_mapping_file_path)
save_id_mapping(item_id_mapping, item_mapping_file_path)

print(f"数据已保存到 {txt_file_path}")
print(f"用户ID映射字典已保存到 {user_mapping_file_path}")
print(f"物品ID映射字典已保存到 {item_mapping_file_path}")