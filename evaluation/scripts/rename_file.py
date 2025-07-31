import os

def rename_files_in_directory(root_dir, old_str="sage_seen", new_str="sage_seen"):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if old_str in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(old_str, new_str)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} → {new_path}")

if __name__ == "__main__":
    target_directory = "/data/leiyu/workspace/PSAlign/evaluation/eval_gpt"  # ← 修改为你要处理的路径
    rename_files_in_directory(target_directory)
