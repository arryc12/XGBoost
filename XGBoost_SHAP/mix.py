# mix_csv_in_folder.py
import pandas as pd
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def pick_folder() -> Path:
    """弹窗选文件夹（如不想弹窗，直接 return Path(r'你的路径') 即可）"""
    root = tk.Tk()
    root.withdraw()
    folder = Path(filedialog.askdirectory(title='选择包含 3 个 CSV 的文件夹'))
    root.destroy()
    return folder

def main(folder: Path | None = None):
    if folder is None:
        folder = pick_folder()

    # 1. 找到文件夹里前 3 个 csv
    csv_files = sorted(folder.glob('*.csv'))[:3]
    if len(csv_files) < 3:
        print(f'❌ {folder} 中 csv 文件不足 3 个！')
        return

    print('即将混合：', [f.name for f in csv_files])

    # 2. 读入并标记来源
    dfs = [pd.read_csv(f) for f in csv_files]

    # 3. 合并 + 随机打乱
    mixed = (
        pd.concat(dfs, ignore_index=True)
          .sample(frac=1, random_state=random.randint(0, 9999))
          .reset_index(drop=True)
    )

    # 4. 输出到同一文件夹
    out_file = folder / 'mixed_output.csv'
    mixed.to_csv(out_file, index=False)
    print(f'✅ 混合完成，共 {len(mixed)} 行，已保存为 {out_file}')

if __name__ == '__main__':
    # 想硬编码路径就把下面注释打开，把路径换成你的
    # main(Path(r'D:\data\my_csvs'))
    main()