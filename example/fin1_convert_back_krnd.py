import torch
import csv
import pandas as pd
import argparse
import os
from tqdm import tqdm

def calculate_average_ratings(interactions_df):
    item_avg_ratings = interactions_df.groupby('item_id')['rating'].mean().to_dict()
    return item_avg_ratings

def main(root_path, train_pth_path, output_pth, reverse_mapping_path):
    inter_csv_path = os.path.join(root_path, 'inter.csv')
    sasrec_output_path = os.path.join(root_path, output_pth)
    ratings_output_path = os.path.join(root_path, 'ratings.csv')
    
    # Load datasets
    train_data = torch.load(train_pth_path)
    interactions_df = pd.read_csv(inter_csv_path)
    reverse_item_mapping = torch.load(reverse_mapping_path)

    # Calculate average ratings
    item_avg_ratings = calculate_average_ratings(interactions_df)

    # Create transition dictionary
    transition_dict = {}
    for user, user_data in tqdm(interactions_df.groupby('user_id')):
        sorted_user_data = user_data.sort_values(by='timestamp')
        for i in range(len(sorted_user_data) - 1):
            prev_item = sorted_user_data.iloc[i]['item_id']
            next_item = sorted_user_data.iloc[i+1]['item_id']
            rating = sorted_user_data.iloc[i+1]['rating']
            timestamp = sorted_user_data.iloc[i+1]['timestamp']
            transition_dict[(prev_item, next_item)] = (rating, timestamp)
    
    max_existing_user_id = interactions_df['user_id'].max() if not interactions_df.empty else 0
    print(max_existing_user_id)
    
    new_user_id = max_existing_user_id + 1
    interaction_dict = {(row['user_id'], row['item_id']): (row['rating'], row['timestamp']) for _, row in interactions_df.iterrows()}
    
    unique_user_data = []
    seen_users = set()
    for data in train_data:
        user_id = data[0]
        if user_id in seen_users and user_id != 1:
            continue
        if user_id == 1 or user_id not in seen_users:
            unique_user_data.append(data)
        seen_users.add(user_id)
    train_data = unique_user_data

    with open(sasrec_output_path, 'w', newline='') as sasrec_csvfile:
        sasrec_writer = csv.writer(sasrec_csvfile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        sasrec_writer.writerow(['index', 'user_id', 'sequence_item_ids', 'sequence_ratings', 'sequence_timestamps'])
        known_users = set(interactions_df['user_id'])
        processed_users = []
        max_item_idx=0
        for index, data in tqdm(enumerate(train_data)):
            user_id, history, target_data, seq_len, label, domain_id = data
            if(max(history)>max_item_idx):
                max_item_idx=max(history)
            if(max(target_data)>max_item_idx):
                max_item_idx=max(target_data)
        for index, data in tqdm(enumerate(train_data)):
            user_id, history, target_data, seq_len, label, domain_id = data
            if seq_len > 3:
                tmp_user_id = user_id
                if (user_id not in known_users) or (user_id in processed_users):
                    tmp_user_id = new_user_id
                    new_user_id += 1
                processed_users.append(user_id)
                user_id = tmp_user_id
                sequence_ratings = []
                sequence_timestamps = []
                max_timestamp = max((interaction_dict.get((user_id, item_id), (None, 0))[1] for item_id in history), default=0)
                filtered_items = []
                flag = 0
                sequence_ratings_in_regen = []
                for idx, item_id in enumerate(history):
                    if item_id >= (max_item_idx-1):
                        rating = item_id - (max_item_idx-1)
                        flag = 1
                        sequence_ratings_in_regen.append(rating)
                    else:
                        if idx > 0:
                            prev_item_id = history[idx - 1]
                            if (prev_item_id, item_id) in transition_dict:
                                rating, timestamp = transition_dict[(prev_item_id, item_id)]
                            else:
                                rating, timestamp = interaction_dict.get((user_id, item_id), (item_avg_ratings.get(item_id, 1), None))
                        else:
                            rating, timestamp = interaction_dict.get((user_id, item_id), (item_avg_ratings.get(item_id, 1), None))
                        label = 1 if rating >= 1 else 0
                        if item_id != 0:
                            filtered_items.append(item_id)
                            sequence_ratings.append(label)
                            if timestamp is None:
                                timestamp = max_timestamp + idx * 50
                            sequence_timestamps.append(timestamp)

                if len(sequence_timestamps) < len(sequence_ratings):
                    sequence_timestamps.extend([max_timestamp + i * 50 for i in range(len(sequence_timestamps), len(sequence_ratings))])
                if len(filtered_items) > 0:
                    sequence_item_ids_str = ",".join(map(str, [reverse_item_mapping.get(item_id, item_id) for item_id in filtered_items]))
                    if flag == 0:
                        sequence_ratings_str = ",".join(map(str, sequence_ratings))
                    else:
                        n = len(sequence_timestamps) - len(sequence_ratings_in_regen)
                        sequence_ratings_in_regen = sequence_ratings_in_regen + sequence_ratings[-n:] if n > 0 else sequence_ratings_in_regen
                        sequence_ratings_str = ",".join(map(str, sequence_ratings_in_regen))
                    sequence_timestamps_str = ",".join(map(str, sequence_timestamps))
                    sasrec_writer.writerow([index, user_id, sequence_item_ids_str, sequence_ratings_str, sequence_timestamps_str])
    print('Conversion to sasrec_format.csv completed!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data with root path.')
    parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
    parser.add_argument('--train_pth_path', type=str, required=True, help='Path to the train data .pth file.')
    parser.add_argument('--output_pth', type=str, required=True, help='Output file path for sasrec_format.csv.')
    parser.add_argument('--reverse_mapping_path', type=str, required=True, help='Path to the reverse item mapping .pth file.')
    args = parser.parse_args()
    main(args.root_path, args.train_pth_path, args.output_pth, args.reverse_mapping_path)