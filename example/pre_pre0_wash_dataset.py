import pandas as pd
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Filter items used less than a threshold.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input sasrec_format.csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output filtered CSV")
    parser.add_argument("--min_item_freq", type=int, default=5, help="Minimum frequency of items to keep")
    return parser.parse_args()

def compute_stats(df, title):
    lengths = df["sequence_item_ids"].apply(lambda x: len(x.split(",")))
    total_interactions = lengths.sum()
    avg_length = lengths.mean()
    print(f"{title}统计信息:")
    print(f"- 总用户数: {len(df)}")
    print(f"- 总交互数: {total_interactions}")
    print(f"- 平均每个用户的交互长度: {avg_length:.2f}")
    print("-" * 40)

def main():
    args = parse_args()

    df = pd.read_csv(args.input_path)

    print("➡️ 原始数据统计")
    compute_stats(df, "原始数据")

    # 统计所有 item 出现频率
    all_items = []
    for seq in df["sequence_item_ids"]:
        all_items.extend(seq.split(","))

    item_counts = Counter(all_items)
    valid_items = {item for item, count in item_counts.items() if count >= args.min_item_freq}

    def filter_sequence(seq_str, rating_str, timestamp_str):
        items = seq_str.split(",")
        ratings = rating_str.split(",")
        timestamps = timestamp_str.split(",")
        
        filtered = [
            (i, r, t)
            for i, r, t in zip(items, ratings, timestamps)
            if i in valid_items
        ]
        if not filtered:
            return "", "", ""
        items_f, ratings_f, timestamps_f = zip(*filtered)
        return ",".join(items_f), ",".join(ratings_f), ",".join(timestamps_f)

    df[["sequence_item_ids", "sequence_ratings", "sequence_timestamps"]] = df.apply(
        lambda row: filter_sequence(row["sequence_item_ids"], row["sequence_ratings"], row["sequence_timestamps"]),
        axis=1,
        result_type="expand"
    )

    # 删除空序列
    df = df[df["sequence_item_ids"] != ""]

    print("✅ 过滤后数据统计")
    compute_stats(df, "过滤后数据")

    df.to_csv(args.output_path, index=False)
    print(f"✅ 过滤后的文件已保存到: {args.output_path}")

if __name__ == "__main__":
    main()
