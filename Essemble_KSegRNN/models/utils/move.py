import os
import shutil

# 更改当前工作目录

os.chdir('/data/hqh/Router/traffic')

# 获取当前工作目录
current_dir = os.getcwd()

print("✅ 当前工作目录:", current_dir)
# 目标目录
target_dir = os.path.join(current_dir, 'input_data')

# 确保 input_data 文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历当前目录下的所有文件
for file in os.listdir(current_dir):
    if file.endswith('.npy'):
        file_path = os.path.join(current_dir, file)
        # 确保不是已经在 input_data 文件夹内的文件
        if os.path.isfile(file_path):
            print(f"📦 Moving {file} → input_data/")
            shutil.move(file_path, target_dir)

print("✅ 所有 .npy 文件已移动至 input_data 文件夹。")
