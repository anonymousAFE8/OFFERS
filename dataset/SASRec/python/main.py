import os
import time
import torch
import argparse
from tqdm import tqdm
from model import SASRec
from utils import *
import json
from datetime import datetime

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--tokenize_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--norm_first', action='store_true', default=False)
parser.add_argument('--tokenizer_threshold', default=0.2, type=float)
parser.add_argument('--tokenizer_threshold_p2', default=0.2, type=float)
parser.add_argument('--output_dataset_name', default='ml-1m', type=str)
parser.add_argument('--target_name', default=False, type=bool)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

def entropy(prob):
    """Calculate the entropy of a single probability."""
    return -prob * np.log2(prob) if prob > 0 else 0

def calculate_token_entropies(sequences, item_idx_seq, model, args):
    """Calculate the entropy for given sequences and return an entropy dictionary."""
    model.eval()
    entropy_dict = {}
    with torch.no_grad():
        for seq_key, next_items in tqdm(sequences.items(), desc="Calculating new entropies", leave=False):
            seq_tensor = np.array([np.array(seq_key)])
            user_id = np.array([1])  # A placeholder for a single user_id; adjust if needed
            # Generate item_idx, e.g., 100 random samples that are not in the current sequence
            item_idx = item_idx_seq  # Use passed item index sequence as reference

            predictions = -model.predict(user_id, seq_tensor, item_idx)
            probs = torch.softmax(predictions, dim=-1)
            entropies = [entropy(probs[0, item-1].item()) for item in next_items]  # Use the first row, single user case
            entropy_dict[seq_key] = entropies
            
    return entropy_dict

def count_sequences_occurrences(seq_key, user_train):
    """Count how many times seq_key appears in the user_train dataset."""
    count = 0
    seq_len = len(seq_key)
    for user, items in user_train.items():
        for i in range(len(items) - seq_len + 1):
            if tuple(items[i:i + seq_len]) == seq_key:
                count += 1
    return count

def build_successor_dict(user_train, seq_len):
    """Build a successor dictionary for sequences of specified length."""
    successor_dict = {}
    for user, items in user_train.items():
        for i in range(len(items) - seq_len):
            current_seq = tuple(items[i:i + seq_len])
            next_item = items[i + seq_len]
            if current_seq not in successor_dict:
                successor_dict[current_seq] = []
            if next_item not in successor_dict[current_seq]:
                successor_dict[current_seq].append(next_item)
    return successor_dict

def build_count_dict(user_train, seq_len):
    """Build a count dictionary for sequences of specified length."""
    count_dict = {}
    for user, items in user_train.items():
        for i in range(len(items) - seq_len + 1):
            current_seq = tuple(items[i:i + seq_len])
            if current_seq not in count_dict:
                count_dict[current_seq] = 0
            count_dict[current_seq] += 1
    return count_dict

def calculate_item_entropy(item_counts, total_count):
    """ 
    计算每个物品的熵值，即 -p * log(p)
    :param item_counts: 物品的出现次数
    :param total_count: 所有物品的总出现次数
    :return: 物品的熵值
    """
    probabilities = item_counts / total_count
    return - np.sum(probabilities * np.log2(probabilities + 1e-9))

def tokenizer(model, dataset, args):
    user_train, _, _, _, itemnum = dataset
    token_set = []
    print('tokens'+args.dataset)
    # Step 1: Prepare all independent item IDs (item index sequence)
    item_idx_seq = list(range(1, itemnum + 1))
    
    # Calculate total occurrences for all items
    total_items_count = sum([len(items) for items in user_train.values()])
    item_counts = np.zeros(itemnum + 1)
    
    # Count occurrences for each item
    for items in user_train.values():
        for item in items:
            item_counts[item] += 1
    
    # Step 2: Calculate entropy for each item
    print("Calculating entropy for each item...")
    item_entropies = {item: calculate_item_entropy(item_counts[item], total_items_count) 
                      for item in range(1, itemnum + 1)}
    
    average_ent = np.mean(list(item_entropies.values()))
    print(f"Global average entropy: {average_ent}")
    average_ent = args.tokenizer_threshold*5*average_ent
    average_ent_p2 = args.tokenizer_threshold_p2*5*average_ent
    
    # Step 3: Filter items based on entropy and store in ToAnalysis_length_pre
    ToAnalysis_length_pre = {}
    for item, ent in item_entropies.items():
        if ent < average_ent and item_counts[item] > 2:
            ToAnalysis_length_pre[(item,)] = ent
    print(len(ToAnalysis_length_pre.keys()))
    # Proceed with processing sequences of increasing length
    iteration = 1
    while ToAnalysis_length_pre:
        seq_len = len(list(ToAnalysis_length_pre.keys())[0])
        print(f"\nIteration {iteration}, analyzing sequences of length {seq_len + 1}...")
        # Build the successor dictionary for the current sequence length
        successor_dict = build_successor_dict(user_train, seq_len)
        
        # Count occurrences of the top 5 sequences in the original dataset
        top_sequences_stats = []
        for idx, (seq_key, seq_entropy) in enumerate(list(ToAnalysis_length_pre.items())[:5]):
            count = count_sequences_occurrences(seq_key, user_train)
            top_sequences_stats.append((seq_key, seq_entropy, count))
        
        print("Top 5 sequences to analyze (with occurrence count):")
        for seq_key, seq_entropy, count in top_sequences_stats:
            print(f"Sequence: {seq_key}, Entropy: {seq_entropy}, Count: {count}")
        
        new_sequences = {}
        for seq_key in ToAnalysis_length_pre.keys():
            if seq_key in successor_dict:
                new_sequences[seq_key] = successor_dict[seq_key]
        
        if not new_sequences:
            break
        
        # Step 6: Compute new entropy for extended sequences
        print(f"Calculating new entropy for extended sequences...")
        new_entropy_dict = calculate_token_entropies(new_sequences, item_idx_seq, model, args)
        
        updated_ToAnalysis_length_pre = {}
        count_dict = build_count_dict(user_train, seq_len + 1)
        count_dict_1 = build_count_dict(user_train, seq_len)
        
        for seq_key, next_entropy in new_entropy_dict.items():
            for idx, each_nxt_ent in enumerate(next_entropy):
                combined_entropy = ToAnalysis_length_pre[seq_key] + each_nxt_ent
                extended_key = seq_key + (new_sequences[seq_key][idx],)
                if combined_entropy < average_ent and len(extended_key) < 5:
                    if count_dict[extended_key] >= 2:
                        updated_ToAnalysis_length_pre[extended_key] = combined_entropy
                    else:
                        if(len(seq_key)>1 and seq_key not in token_set):
                            token_set.append(seq_key)
                else:
                    if count_dict_1[seq_key] >= 2:
                        if(len(seq_key)>1 and seq_key not in token_set):
                            token_set.append(seq_key)
        print(len(updated_ToAnalysis_length_pre.keys()),len(token_set))
        ToAnalysis_length_pre = updated_ToAnalysis_length_pre
        iteration += 1

    # Step 10: 输出 Token 集合到 JSON 文件
    if args.target_name:
        with open(args.output_dataset_name+'.json', 'w') as f:
            json.dump(token_set, f, indent=1)
    else:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(args.dataset+'_token_T_'+str(args.tokenizer_threshold)+'_T2_'+str(args.tokenizer_threshold_p2)+'_'+current_time+'.json', 'w') as f:
            json.dump(token_set, f, indent=1)
    print("Tokenization completed. Tokens written to token.json")


if __name__ == '__main__':

    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.tokenize_only:
    # 假设此处已经有训练后的模型 `model` 和数据 `dataset`
        model.eval()  # 确保模型处于评估模式
        tokenizer(model, dataset, args)
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    if not (args.inference_only or args.tokenize_only):
        bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

        best_val_ndcg, best_val_hr = 0.0, 0.0
        best_test_ndcg, best_test_hr = 0.0, 0.0
        T = 0.0
        t0 = time.time()
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only: break # just to decrease identition
            for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            if epoch % 20 == 0:
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end='')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                        % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                    best_val_ndcg = max(t_valid[0], best_val_ndcg)
                    best_val_hr = max(t_valid[1], best_val_hr)
                    best_test_ndcg = max(t_test[0], best_test_ndcg)
                    best_test_hr = max(t_test[1], best_test_hr)
                    folder = args.dataset + '_' + args.train_dir
                    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                    torch.save(model.state_dict(), os.path.join(folder, fname))

                f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
                model.train()
        
            if epoch == args.num_epochs:
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))
        
        f.close()
        sampler.close()
        print("Done")
#Step 0:设置Token集合为[]
#Step 1:记录数据集中所有[item_a,item_b]的序列，以key为item_a,value为item_b为字典记录
#Step 2:以batchsize计算[item_a，item_b]中item_b的生成概率，计算序列熵p(item_b)log(p(item_b))
#Step 3:输出所有[item_a，item_b]的序列熵平均值，记录该平均值为ent(全局只记录这一次)，输出
#Step 4:筛选平均值低于ent的[item_a,item_b]的序列，记录为字典ToAnalysis_length_pre，key为[item_a,item_b],value为其序列熵
#其余序列加入Token集合
#Step 5:记录数据集中所有[item_a,item_b,item_c]的序列，以key为[item_a,item_b],value为item_c处理
#Step 6:计算[item_a,item_b,item_c]的序列熵，其为ToAnalysis_length_pre中[item_a,item_b]的值加上p(item_c)log(p(item_c)),记录为ToAnalysis_length_pre_new
#Step 6.5:ToAnalysis_length_pre_new=ToAnalysis_length_pre
#Step 7:选取[item_a,item_b,item_c]序列熵低于ent的序列进入下一步，其余加入Token集合
#Step 8:记录数据集中所有[item_a,item_b,item_c,item_d]的序列，以key为[item_a,item_b,item_c],value为item_d处理
#Step 9:如此循环，直到没有任何序列熵高于ent
#Step 10:输出Token集合