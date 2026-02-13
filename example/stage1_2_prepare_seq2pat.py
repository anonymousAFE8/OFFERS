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

def process_chunk(data_chunk, args):
    def worker(data_chunk, conn, args):
        try:
            seq2pat = Seq2Pat(sequences=data_chunk, n_jobs=args.n_jobs, max_span=args.alpha)
            patterns = seq2pat.get_patterns(min_frequency=args.beta)
            patterns_value = [_[:-1] for _ in patterns]
            patterns_value = [lst_ for lst_ in patterns_value if len(lst_) >= 3]
            conn.send(patterns_value)
        except Exception as e:
            conn.send(e)

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=worker, args=(data_chunk, child_conn, args))
    p.start()

    # 设置最大执行时间（秒）
    timeout = 600

    if parent_conn.poll(timeout):
        result = parent_conn.recv()
        p.join()  # 确保子进程结束
        if isinstance(result, Exception):
            raise result  # 如果是异常则抛出
        return result
    else:
        p.terminate()  # 超时则终止子进程
        p.join()  # 确保子进程结束
        print(f"Process timed out. Splitting data chunk {len(data_chunk)} into smaller parts.")
        mid_point = len(data_chunk) // 2
        
        # 递归地处理分块
        first_half_patterns = process_chunk(data_chunk[:mid_point], args)
        second_half_patterns = process_chunk(data_chunk[mid_point:], args)
        
        return first_half_patterns + second_half_patterns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='The path to the dataset.')
    parser.add_argument('--chunk_index', type=int, required=True, help='Which chunk index to process.')
    parser.add_argument('--alpha', type=int, default=5, help='The sliding window size for pre-training dataset construction.')
    parser.add_argument('--beta', type=int, default=10, help='The threshold for pre-training dataset construction.')
    parser.add_argument('--n_jobs', type=int, default=10, help='The job number for Seq2Pat pattern mining.')
    parser.add_argument('--max_seq_len', type=int, default=50, help='The maximum sequence length.')
    args = parser.parse_args()
    
    chunk_file = os.path.join(args.root_path, 'chunks', f'seq2pat_data_chunk_{args.chunk_index}.pth')
    data_chunk = torch.load(chunk_file)
    print(f'Processing data chunk with size {len(data_chunk)}...')
    
    patterns_value = process_chunk(data_chunk, args)

    output_chunk_path = os.path.join(args.root_path, f'patterns_value_chunk_{args.chunk_index}.pth')
    torch.save(patterns_value, output_chunk_path)
    print(f'Saved patterns for chunk {args.chunk_index} to {output_chunk_path}')