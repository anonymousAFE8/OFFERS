import json
import argparse
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def reverse_mapping(mapping):
    """ Create a reverse mapping from value to key """
    return {v: k for k, v in mapping.items()}

def restore_patterns(tokens, reverse_item_mapping):
    """ Restore item IDs in the tokens using the reverse mapping """
    restored_patterns = []
    for pattern in tokens:
        restored_pattern = [reverse_item_mapping.get(item_id, item_id) for item_id in pattern]
        restored_patterns.append(restored_pattern)
    return restored_patterns

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

parser = argparse.ArgumentParser(description='Process data with root path.')
parser.add_argument('--root_path', type=str, required=True, help='The root directory path for reading and writing files.')
args = parser.parse_args()
# 示例使用
item_mapping_file_path = args.root_path+'/'+args.root_path.split('/')[-3]+'_item_id_mapping.json'  # 请替换为实际文件路径
tokens_file_path =  args.root_path+'/'+args.root_path.split('/')[-3]+'.json'  # 请替换为实际文件路径
restored_tokens_file_path =  args.root_path+'/restored_tokens.json'  # 输出的还原文件

# 加载映射文件和token文件
item_id_mapping = load_json(item_mapping_file_path)
tokens = load_json(tokens_file_path)

# 创建反向映射字典
reverse_item_id_mapping = reverse_mapping(item_id_mapping)

# 还原patterns中的ID
restored_patterns = restore_patterns(tokens, reverse_item_id_mapping)

# 保存还原后的patterns
save_json(restored_patterns, restored_tokens_file_path)

print(f"还原后的tokens已保存到 {restored_tokens_file_path}")