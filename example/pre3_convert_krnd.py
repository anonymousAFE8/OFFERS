import pandas as pd
import os
import torch
import argparse

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description="Process data with root_path.")
parser.add_argument('--root_path', type=str, required=True, help='Root path for input and output files')
parser.add_argument('--max_seq_len', type=int, required=True, help='Root path for input and output files')
args = parser.parse_args()

# 使用 root_path 参数
root_path = args.root_path
input_filename = os.path.join(root_path, 'sasrec_format.csv')
output_path = os.path.join(root_path)

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 设置其他参数
user_threshold = 5
item_threshold = 5
max_seq_len = args.max_seq_len
PAD = 0  # 用于序列的填充

# 读取CSV文件并解析列
dataset = pd.read_csv(input_filename)
# 假设CSV列格式：index, user_id, sequence_item_ids, sequence_ratings, sequence_timestamps
dataset['sequence_item_ids'] = dataset['sequence_item_ids'].apply(lambda x: list(map(int, x.split(','))))
dataset['sequence_ratings'] = dataset['sequence_ratings'].apply(lambda x: list(map(float, x.split(','))))
dataset['sequence_timestamps'] = dataset['sequence_timestamps'].apply(lambda x: list(map(int, x.split(','))))
# 将sequence_ratings转换为domain_id
dataset['sequence_domain_ids'] = dataset['sequence_ratings'].apply(lambda x: [1 if rating > 0 else 0 for rating in x])

# 展开序列到每个交互记录
records = []
for _, row in dataset.iterrows():
    for (item, rating, timestamp, domain_id) in zip(row['sequence_item_ids'], row['sequence_ratings'], row['sequence_timestamps'], row['sequence_domain_ids']):
        rating = int(rating)
        records.append([row['user_id'], item, rating, timestamp, domain_id])

# 创建DataFrame用于进一步处理
unrolled_df = pd.DataFrame(records, columns=['user_id', 'item_id', 'rating', 'timestamp', 'domain_id'])

# 数据过滤
while True:
    ori_len = len(unrolled_df)
    unrolled_df = unrolled_df[unrolled_df['user_id'].map(unrolled_df['user_id'].value_counts()) >= user_threshold]
    unrolled_df = unrolled_df[unrolled_df['item_id'].map(unrolled_df['item_id'].value_counts()) >= item_threshold]
    if len(unrolled_df) == ori_len:
        break

# 映射用户和物品到新的ID，并构建反向映射
all_user = unrolled_df.user_id
all_item = unrolled_df.item_id
user_id, user_token = pd.factorize(all_user)
item_id, item_token = pd.factorize(all_item)
num_users = len(user_token) + 1  # 0 ID 用于填充
num_items = len(item_token) + 1  # 0 ID 用于填充
user_mapping_dict = {orig_id: new_id + 1 for new_id, orig_id in enumerate(user_token)}  # 从1开始
item_mapping_dict = {orig_id: new_id + 1 for new_id, orig_id in enumerate(item_token)}  # 从1开始

# 反向映射
reverse_user_mapping_dict = {v: k for k, v in user_mapping_dict.items()}
reverse_item_mapping_dict = {v: k for k, v in item_mapping_dict.items()}

unrolled_df['user_id'] = unrolled_df['user_id'].apply(lambda x: user_mapping_dict[x])
unrolled_df['item_id'] = unrolled_df['item_id'].apply(lambda x: item_mapping_dict[x])

# 选择inter.csv需要的列
inter_df = unrolled_df[['user_id', 'item_id', 'rating', 'timestamp', 'domain_id']]
# 保存inter.csv
inter_csv_path = os.path.join(output_path, 'inter.csv')
inter_df.to_csv(inter_csv_path, index=False)
print('inter.csv 生成完成！')

# 分组并处理每个用户序列
user_group = unrolled_df.groupby('user_id')['item_id'].apply(list)
domain_group = unrolled_df.groupby('user_id')['domain_id'].apply(list)

# 生成和保存 seq2pat_data.pth
seq2pat_data = user_group.tolist()
pattern_out_path = os.path.join(output_path, 'seq2pat_data.pth')
torch.save(seq2pat_data, pattern_out_path)
print('seq2pat_data.pth 生成完成！')

# 定义截断或填充函数
def truncate_or_pad(seq, domain_seq=None):
    cur_seq_len = len(seq)
    if domain_seq is not None:
        if cur_seq_len > max_seq_len:
            return seq[-max_seq_len:], domain_seq[-max_seq_len:], max_seq_len
        else:
            return seq + [PAD] * (max_seq_len - cur_seq_len), domain_seq + [0] * (max_seq_len - cur_seq_len), cur_seq_len
    else:
        if cur_seq_len > max_seq_len:
            return seq[-max_seq_len:], max_seq_len
        else:
            return seq + [PAD] * (max_seq_len - cur_seq_len), cur_seq_len

train, val, test, train_wod= [], [], [],[]
for user_id, (user_seq, user_domain_seq) in list(zip(user_group.index, zip(user_group.tolist(), domain_group.tolist()))):
    user_seq = user_seq[-max_seq_len:]  # 保证序列最大长度不超过设定的max_seq_len
    user_domain_seq = user_domain_seq[-max_seq_len:]
    
    # test sample
    if(len(user_seq)>1):
        history, domain_history, seq_len = truncate_or_pad(user_seq[:-1], user_domain_seq[:-1])
        target_data = user_seq[-1]
        target_domain = user_domain_seq[-1]
        label = 1
        test.append([user_id, history, target_data, seq_len, label,  domain_history, history])
    
    # val sample
    if(len(user_seq)>2):
        history, domain_history, seq_len = truncate_or_pad(user_seq[:-2], user_domain_seq[:-2])
        target_data = user_seq[-2]
        target_domain = user_domain_seq[-2]
        label = 1
        val.append([user_id, history, target_data, seq_len, label, domain_history, history])
    
    # train sample
    if(len(user_seq)>3):
        train_history, train_domain_history, seq_len = truncate_or_pad(user_seq[:-3], user_domain_seq[:-3])
        target_data, target_len ,_= truncate_or_pad(user_seq[-seq_len-2:-2], user_domain_seq[-seq_len-2:-2])
        label = [1] * seq_len + [PAD] * (max_seq_len - seq_len)
        train.append([user_id, train_history, target_data, seq_len, label, train_domain_history])

    # train w/o domain sample
    if(len(user_seq)>3):
        train_history, train_domain_history, seq_len = truncate_or_pad(user_seq[:-3], user_domain_seq[:-3])
        target_data, target_len ,_= truncate_or_pad(user_seq[-seq_len-2:-2], user_domain_seq[-seq_len-2:-2])
        label = [1] * seq_len + [PAD] * (max_seq_len - seq_len)
        train_wod.append([user_id, train_history, target_data, seq_len, label, [0]*max_seq_len])

# 保存数据集到文件
torch.save(train, os.path.join(output_path, 'train.pth'))
torch.save(train, os.path.join(output_path, 'train_ori.pth'))
torch.save(train_wod, os.path.join(output_path, 'train_wod.pth'))
torch.save(val, os.path.join(output_path, 'val.pth'))
torch.save(test, os.path.join(output_path, 'test.pth'))

# 保存反向映射
torch.save(reverse_user_mapping_dict, os.path.join(output_path, 'reverse_user_mapping.pth'))
torch.save(reverse_item_mapping_dict, os.path.join(output_path, 'reverse_item_mapping.pth'))

# 输出用户和物品的唯一ID数量
unique_user_ids = unrolled_df['user_id'].unique()
unique_item_ids = unrolled_df['item_id'].unique()
print(f"Unique user IDs: {len(unique_user_ids)}")
print(f"Unique item IDs: {len(unique_item_ids)}")
print('数据处理完成！')