import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import argparse

pattern_dict = {}
pat_key_dict = {}

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
        
class TrieNode:
    def __init__(self):
        self.children = {}
        self.pattern_info = None  # To store pattern and additional info

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, pattern, pattern_info):
        node = self.root
        for item in pattern:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.pattern_info = pattern_info

    def search(self, sequence):
        matches = []
        for i in range(len(sequence)):
            node = self.root
            j = i
            while j < len(sequence) and sequence[j] in node.children:
                node = node.children[sequence[j]]
                if node.pattern_info is not None:
                    # If a pattern ends here, we add it to matches
                    matches.append((i, node.pattern_info))
                j += 1
        return matches

def pattern_exists(target_pattern, target_sequence_ratings):
    """
    检查一个给定的 pattern 和 sequence_ratings 是否存在于 pattern_dict 中。
    
    :param pattern_dict: 存储 pattern 信息的字典
    :param target_pattern: 需要检查的 pattern 列表
    :param target_sequence_ratings: 需要检查的 sequence_ratings 列表
    :return: 如果存在返回 True，否则返回 False
    """
    for value in pattern_dict.values():
        if value['pattern'] == target_pattern and value['sequence_ratings'] == target_sequence_ratings:
            return True
    return False


def match_and_replace_pattern(sequence_items, sequence_ratings, sequence_timestamps, trie, new_id_start):
    processed_sequence = list(sequence_items)
    matches = trie.search(processed_sequence)
    #print(matches)
    
    # 排序匹配项按位置和模式长度
    matches.sort(key=lambda x: (x[0], -len(x[1]["pattern"])))
    
    # 用来存放不重叠的匹配项
    non_overlapping_matches = []

    last_pos = -1

    # 过滤掉重叠的匹配项
    for pos, pattern_info in matches:
        pat_len = len(pattern_info["pattern"])
        if pos >= last_pos:
            non_overlapping_matches.append((pos, pattern_info))
            last_pos = pos + pat_len
    non_overlapping_matches.reverse()
    #print(non_overlapping_matches)
    for pos, pattern_info in non_overlapping_matches:
        pat_key = (tuple(pattern_info["pattern"]), tuple(pattern_info["sequence_ratings"]))
        pat_len = len(pattern_info["pattern"])
        
        if pat_key not in pat_key_dict:
            if not pattern_dict:
                pat_key_dict[pat_key] = new_id_start
                pattern_dict[new_id_start] = {
                    'pattern': processed_sequence[pos:(pos + pat_len)],
                    'sequence_ratings': sequence_ratings[pos:(pos + pat_len)],
                    'sequence_timestamps': sequence_timestamps[pos:(pos + pat_len)]
                }
            else:
                max_pd_id = max(pattern_dict.keys()) + 1
                pat_key_dict[pat_key] = max_pd_id
                pattern_dict[max_pd_id] = {
                    'pattern': processed_sequence[pos:(pos + pat_len)],
                    'sequence_ratings': sequence_ratings[pos:(pos + pat_len)],
                    'sequence_timestamps': sequence_timestamps[pos:(pos + pat_len)]
                }
                #print(pat_key,max_pd_id,pattern_dict[max_pd_id])

        # Replace matched pattern with new ID
        #print(sequence_ratings[pos:(pos + pat_len)],sequence_timestamps[pos:(pos + pat_len)])
        processed_sequence = (processed_sequence[:pos] 
                              + [pat_key_dict[pat_key]] 
                              + processed_sequence[pos + pat_len:])
        sequence_ratings = (sequence_ratings[:pos] 
                            + [int(np.mean(sequence_ratings[pos:(pos + pat_len)]))] 
                            + sequence_ratings[pos + pat_len:])
        sequence_timestamps = (sequence_timestamps[:pos] 
                               + [int(np.mean(sequence_timestamps[pos:(pos + pat_len)]))] 
                               + sequence_timestamps[pos + pat_len:])
    
    return processed_sequence, sequence_ratings, sequence_timestamps

def main():
    # 文件路径
    parser = argparse.ArgumentParser(description='Process data with root path.')
    parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
    args = parser.parse_args()
    tokens_file_path = args.root_path+'restored_tokens.json'
    csv_input_path = args.root_path+'sasrec_format.csv'
    csv_output_path = args.root_path+'sasrec_format.csv'
    pattern_dict_path = args.root_path+'pattern_mappings.json'
    # 加载tokens
    patterns = load_json(tokens_file_path)

    # 构建Trie树并插入模式
    trie = Trie()
    for pattern in patterns:
        # 假设默认 pattern_info
        pattern_int=[]
        for each_token in pattern:
            pattern_int.append(int(each_token))
        pattern_info = {
            "pattern": pattern_int,
            "sequence_ratings": [0] * len(pattern),  # Replace with actual if available
            "sequence_timestamps": [0] * len(pattern)  # Replace with actual if available
        }
        trie.insert(pattern_int, pattern_info)

    # 读取CSV数据为DataFrame
    df = pd.read_csv(csv_input_path)

    # 准备新数据集和字典存储
    new_data = []
    new_id_start = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing sequences'):
        sequence_items = list(map(int, row['sequence_item_ids'].split(',')))
        if(max(sequence_items)>new_id_start):
            new_id_start=max(sequence_items)
    new_id_start=new_id_start+1

    # 使用tqdm显示进度
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing sequences'):
        sequence_items = list(map(int, row['sequence_item_ids'].split(',')))
        ratings = list(map(int, row['sequence_ratings'].split(',')))
        timestamps = list(map(int, row['sequence_timestamps'].split(',')))
        #print(sequence_items)
        processed_sequence, processed_ratings, processed_timestamps = match_and_replace_pattern(sequence_items,ratings, timestamps, trie,new_id_start)
        #print(pattern_dict.keys())
        #print(processed_sequence, processed_ratings, processed_timestamps)
        #input()



        # 更新原始行数据
        new_row = {
            'index': row['index'],
            'user_id': row['user_id'],
            'sequence_item_ids': ','.join(map(str, processed_sequence)),
            'sequence_ratings': ','.join(map(str, processed_ratings)),
            'sequence_timestamps': ','.join(map(str, processed_timestamps)),
        }

        new_data.append(new_row)



    # 保存到新CSV和字典文件
    print(new_data[0])
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(csv_output_path, index=False)

    with open(pattern_dict_path, 'w') as f:
        json.dump(pattern_dict, f, indent=4)

if __name__ == "__main__":
    main()
