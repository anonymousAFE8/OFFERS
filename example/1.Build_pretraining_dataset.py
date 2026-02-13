import os
import torch
import argparse
from tqdm import tqdm
from random import shuffle
from sequential.seq2pat import Seq2Pat
from multiprocessing import Pool, cpu_count

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
    parser.add_argument('--root_path', type=str, default='./dataset/amazon-toys/toy/', help='The path to the training dataset.')
    parser.add_argument('--alpha', type=int, default=5, help='The sliding window size for pre-training dataset construction.')
    parser.add_argument('--beta', type=int, default=4 , help='The threshold for pre-training dataset construction.')
    parser.add_argument('--n_jobs', type=int, default=20 , help='The job number for Seq2Pat pattern mining.')
    parser.add_argument('--max_seq_len', type=int, default=50 , help='The job number for Seq2Pat pattern mining.')
    args = parser.parse_args()
    
    # Some constant configs
    max_seq_len = args.max_seq_len
    
    # Load the original dataset
    seq2pat_data_path = os.path.join(args.root_path, 'seq2pat_data.pth')
    seq2pat_data = torch.load(seq2pat_data_path)
    print(f'Original dataset loaded with size {len(seq2pat_data)}')
    
    seq2pat = Seq2Pat(sequences=seq2pat_data, n_jobs=args.n_jobs, max_span=args.alpha)
    print('Performing rule-based pattern-mining!')
    patterns = seq2pat.get_patterns(min_frequency=args.beta)
    patterns_value = [_[:-1] for _ in patterns]
    patterns_value = [lst_ for lst_ in patterns_value if len(lst_) >= 3]
    print(patterns_value[0:5])
    print(f'Rule-based patterns mined with size {len(patterns_value)}')
    
    original_train_path = os.path.join(args.root_path, 'train_ori.pth')
    original_train = torch.load(original_train_path)
    
    seq_list_ori_domain = extract_seq_domain_info(original_train)
    
    # Process sequences with their domains
    with Pool(cpu_count()) as pool:
        results = []
        for seq_ori, ori_domain in tqdm(seq_list_ori_domain):
            result = pool.apply_async(process_sequence, (seq_ori, patterns_value, ori_domain))
            results.append(result)
        
        data_generation_pair = []
        for result in tqdm(results):
            data_generation_pair.extend(result.get())
    
    print(f'Building sequence-pattern pair dataset with size {len(data_generation_pair)}.')
    print(data_generation_pair[0:5])
    output_path = os.path.join(args.root_path, 'seq-pat-pair.pth')
    torch.save(data_generation_pair, output_path)