import os
import torch
import argparse
from tqdm import tqdm
from random import shuffle
from sequential.seq2pat import Seq2Pat
from multiprocessing import Pool, cpu_count
import gc
def split_data(seq2pat_data_path, chunk_size, dataset,output_dir,max_seq_len):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    seq2pat_data = torch.load(seq2pat_data_path)
    if(dataset=='books'):
        print(dataset)
        seq2pat_data=seq2pat_data[:][-max_seq_len:]
    num_chunks = (len(seq2pat_data) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(seq2pat_data))
        chunk_data = seq2pat_data[start_idx:end_idx]
        
        chunk_path = os.path.join(output_dir, f'seq2pat_data_chunk_{i}.pth')
        torch.save(chunk_data, chunk_path)
        print(f'Saved chunk {i} with size {len(chunk_data)} to {chunk_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True, help='The path to the dataset.')
    parser.add_argument('--chunk_size', type=int, required=True, help='The size of each chunk.')
    parser.add_argument('--max_seq_len', type=int, required=True, help='The size of each chunk.')
    parser.add_argument('--dataset', type=str, required=True, help='The size of each chunk.')
    args = parser.parse_args()

    seq2pat_data_path = os.path.join(args.root_path, 'seq2pat_data.pth')
    output_dir = os.path.join(args.root_path, 'chunks')
    split_data(seq2pat_data_path, args.chunk_size, args.dataset,output_dir,args.max_seq_len)