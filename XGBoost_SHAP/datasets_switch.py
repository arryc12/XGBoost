import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ========== 1. 参数 ==========
root_dir  = Path(r'data\erect\data0\bubble_flow')  # 要扫的文件夹
out_file  = Path(r'XGBoost_SHAP\datasets\bubble.csv')              # 最终汇总表
chunk_size= 2000
col_idx   = 4                                       # 第 5 列

# ========== 2. 工具函数 ==========
def stats_group(arr):
    return {'mean': arr.mean(),
            'std' : arr.std(ddof=0),
            'skew': stats.skew(arr, bias=False),
            'kurt': stats.kurtosis(arr, bias=False)}

# ========== 3. 主循环 ==========
all_res = []
for csv_path in root_dir.rglob('*.csv'):   # 递归扫所有 csv
    try:
        # 读第 5 列并清掉空值
        s = pd.read_csv(csv_path, usecols=[col_idx], skiprows=1, header=None, engine='c').iloc[:, 0].dropna()
        # 切片统计
        for start in range(0, len(s), chunk_size):
            piece = s.iloc[start:start+chunk_size]
            row = stats_group(piece.to_numpy())
            all_res.append(row)
    except Exception as e:
        print(f'跳过文件 {csv_path} ：{e}')

# ========== 4. 输出 ==========
if all_res:
    df = pd.DataFrame(all_res)
    df['label'] = 1
    df.to_csv(out_file, index=False, float_format='%.6f')
    print(f'全部完成！共 {len(df)} 组，结果见 {out_file}')
else:
    print('目录下未找到可处理的 csv。')