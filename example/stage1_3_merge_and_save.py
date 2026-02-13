import os
import torch
import argparse
from tqdm import tqdm
from random import shuffle
from sequential.seq2pat import Seq2Pat
from multiprocessing import Pool, cpu_count
import gc
# Existing helper functions such as process_sequence, find_subsequence_indices, etc. remain unchanged...
def is_sublist(sublst, lst):
    for element in sublst:
        try:
            ind = lst.index(element)
        except ValueError:
            return False
        lst = lst[ind+1:]
    return True

def find_subsequence_indices(pattern, seq_ori):
    """
    返回 seq_ori 中的索引列表，使得这些索引上的元素按顺序构成 pattern。
    如果无法找到完整的 pattern 作为子序列，返回 None。
    """
    indices = []
    iter_seq_ori = iter(enumerate(seq_ori))
    
    try:
        for elem in pattern:
            for index, value in iter_seq_ori:
                if value == elem:
                    indices.append(index)
                    break
        return indices
    except StopIteration:
        return None

def process_sequence(seq_ori, patterns_value, ori_domain):
    local_data_generation_pair = []
    shuffle(patterns_value)
    cnt = 0
    for pattern in patterns_value:
        if cnt >= 10:
            break
        indices = find_subsequence_indices(pattern, seq_ori)
        if indices is not None:  # 确保 pattern 是 seq_ori 的子序列
            if len(indices) == len(pattern):
                # 获取 pattern 对应的位置的 ori_domain 部分
                pattern_domain = [ori_domain[i] for i in indices]
                local_data_generation_pair.append([seq_ori, pattern, ori_domain, pattern_domain])
                cnt += 1
    return local_data_generation_pair

def extract_seq_domain_info(original_train):
    seq_list_ori_domain = []
    for item in original_train:
        seq = item[1][:item[3]] + [item[2][item[3] - 1]]
        domain = item[5]  # 根据示例中的格式，第六个元素是领域信息
        seq_list_ori_domain.append((seq, domain))
    return seq_list_ori_domain

def truncate_or_pad(seq, max_seq_len):
    cur_seq_len = len(seq)
    if cur_seq_len > max_seq_len:
        return seq[-max_seq_len:]
    else:
        return seq + [0] * (max_seq_len - cur_seq_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='The path to the dataset.')
    parser.add_argument('--num_chunks', type=int, required=True, help='Total number of chunks.')
    args = parser.parse_args()

    all_patterns_value = []
    for i in range(args.num_chunks):
        chunk_path = os.path.join(args.root_path, f'patterns_value_chunk_{i}.pth')
        patterns_value = torch.load(chunk_path)
        all_patterns_value.extend(patterns_value)
        print(f'Merged patterns from chunk {i}')

    original_train_path = os.path.join(args.root_path, 'train_ori.pth')
    original_train = torch.load(original_train_path)
    
    seq_list_ori_domain = extract_seq_domain_info(original_train)
    
    # Process sequences with their domains
    with Pool(cpu_count()) as pool:
        results = []
        for seq_ori, ori_domain in tqdm(seq_list_ori_domain):
            result = pool.apply_async(process_sequence, (seq_ori, all_patterns_value, ori_domain))
            results.append(result)
        
        data_generation_pair = []
        for result in tqdm(results):
            data_generation_pair.extend(result.get())
    
    print(f'Building sequence-pattern pair dataset with size {len(data_generation_pair)}.')
    print(data_generation_pair[0:5])
    output_path = os.path.join(args.root_path, 'seq-pat-pair.pth')
    torch.save(data_generation_pair, output_path)