import os
import pandas as pd

def print_csv_shapes(folder_path):
    # 获取文件夹下所有CSV文件的路径
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    
    if not csv_files:
        print("文件夹中没有CSV文件。")
        return
    
    # 遍历所有CSV文件并打印它们的形状
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        print(f"{file}: {data.shape}")

# 测试
folder_path = './Raw Data/Add Xe/Core'  # 替换为你的文件夹路径

print_csv_shapes(folder_path)
