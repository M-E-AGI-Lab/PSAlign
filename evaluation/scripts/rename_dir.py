import os

def rename_subdirs(root_dir: str):
    """
    将root_dir下所有子目录中名称包含'user_adapter'的，重命名为将'user_adapter'替换为'psa'的名称
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # 使用复制列表避免修改时影响迭代
        original_dirs = dirnames[:]
        for dirname in original_dirs:
            if 'sage_seen' in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_name = dirname.replace('sage_seen', 'sage_seen')
                new_path = os.path.join(dirpath, new_name)

                print(f"Renaming: {old_path} → {new_path}")
                os.rename(old_path, new_path)

                # 修改原始 dirnames，防止 os.walk 进入重命名后的目录路径出错
                dirnames[dirnames.index(dirname)] = new_name

if __name__ == "__main__":
    target_directory = "/data/leiyu/workspace/PSAlign/evaluation/Q16"  # ← 替换为你的路径
    rename_subdirs(target_directory)
