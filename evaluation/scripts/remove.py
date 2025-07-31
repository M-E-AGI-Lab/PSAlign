import os
import torch

def rename_keys_in_bin(file_path, old_key, new_key):
    """
    重命名bin文件中的字典键名
    :param file_path: 文件路径
    :param old_key: 原键名
    :param new_key: 新键名
    """
    try:
        # 加载bin文件
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        
        # 检查是否是字典类型
        if not isinstance(state_dict, dict):
            print(f"跳过非字典文件: {file_path}")
            return
        
        # 如果存在旧键则重命名
        if old_key in state_dict:
            # 弹出旧键值对并添加新键值对
            state_dict[new_key] = state_dict.pop(old_key)
            # 保存回原路径
            torch.save(state_dict, file_path)
            print(f"已处理: {file_path}")
        else:
            print(f"未找到键 '{old_key}': {file_path}")
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")

def process_directory(root_dir, old_key, new_key):
    """
    遍历目录下所有bin文件并处理
    :param root_dir: 根目录
    :param old_key: 原键名
    :param new_key: 新键名
    """
    # 遍历目录及其子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 只处理bin文件
            if filename.endswith('.bin'):
                file_path = os.path.join(dirpath, filename)
                rename_keys_in_bin(file_path, old_key, new_key)

if __name__ == "__main__":
    # 配置参数
    ROOT_DIRECTORY = "../trained_models"  # 替换为你的目录路径
    OLD_KEY = "user_adapter"             # 要替换的旧键名
    NEW_KEY = "psa_adapter"              # 新键名
    
    # 执行处理
    print(f"开始处理目录: {ROOT_DIRECTORY}")
    print(f"将键名 '{OLD_KEY}' 替换为 '{NEW_KEY}'")
    process_directory(ROOT_DIRECTORY, OLD_KEY, NEW_KEY)
    print("处理完成")
