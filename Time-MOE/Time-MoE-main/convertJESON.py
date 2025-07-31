# #读取./datasets下每一个.csv结尾文件的前60%行，对于每个csv，得到一个[时间长度,变量]的二维矩阵，保存成一个jsonl，里面数据{"sequence": [1.0, 2.0, 3.0, ...]}
# {"sequence": [11.0, 22.0, 33.0, ...]}，其中共有变量数量个sequence
import os
import pandas as pd
import json

input_dir = './dataset'
output_dir = './jsonl_outputs'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith('.csv'):
        
        fpath = os.path.join(input_dir, fname)
        df = pd.read_csv(fpath)
        # 🔧 只保留数值型列（跳过第一列时间戳或其他文本列）
        df = df.select_dtypes(include=['number'])
        n_rows = int(len(df) * 0.6)
        df = df.iloc[:n_rows]  # 前 60%

        data = df.to_numpy()  # shape: [time_len, num_vars]
        time_len, num_vars = data.shape

        # 输出文件名同名（只后缀改为.jsonl）
        out_name = os.path.splitext(fname)[0] + '.jsonl'
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, 'w') as fout:
            for var_idx in range(num_vars):
                sequence = data[:, var_idx].tolist()
                fout.write(json.dumps({"sequence": sequence}) + '\n')
