"""
功能函数模块，负责读取不同格式的文件并返回摘要信息。
支持格式：.tdms, .csv, .xls, .xlsx
"""

import os
import pandas as pd
import numpy as np

def process_files(file_paths):
    """
    处理多个文件，返回每个文件的摘要信息列表。

    Args:
        file_paths (list): 文件路径列表

    Returns:
        list: 每个文件对应的摘要字符串
    """
    results = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.tdms':
                info = read_tdms(file_path)
            elif ext == '.csv':
                info = read_csv(file_path)
            elif ext in ('.xls', '.xlsx'):
                info = read_excel(file_path)
            else:
                info = f"不支持的文件格式: {ext}"
        except Exception as e:
            info = f"读取失败: {str(e)}"
        results.append(f"{os.path.basename(file_path)}: {info}")
    return results


def read_tdms(file_path):
    """读取TDMS文件，返回数据摘要"""
    try:
        from nptdms import TdmsFile
        with TdmsFile.open(file_path) as tdms_file:
            groups = tdms_file.groups()
            summary = f"TDMS文件，包含 {len(groups)} 个组"
            # 可添加更详细的组信息，此处简化
            return summary
    except ImportError:
        return "需要安装npTDMS库"


def read_csv(file_path):
    """读取CSV文件，返回数据摘要"""
    try:
        df = pd.read_csv(file_path)
        return f"CSV文件，形状: {df.shape}"
    except Exception as e:
        return f"CSV读取错误: {e}"


def read_excel(file_path):
    """读取Excel文件，返回数据摘要"""
    try:
        # 读取第一个工作表
        df = pd.read_excel(file_path, sheet_name=0)
        return f"Excel文件，形状: {df.shape}"
    except Exception as e:
        return f"Excel读取错误: {e}"
    

def load_data(file_path):
    """
    读取文件并返回完整的 pandas DataFrame。
    对于 TDMS 文件，尝试将第一个组的所有通道合并为 DataFrame。
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.tdms':
            return load_tdms_as_dataframe(file_path)
        elif ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ('.xls', '.xlsx'):
            return pd.read_excel(file_path, sheet_name=0)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    except Exception as e:
        raise RuntimeError(f"加载文件失败: {str(e)}")

def load_tdms_as_dataframe(file_path):
    """
    将 TDMS 文件转换为 DataFrame。
    假设文件至少包含一个组，将该组的所有通道合并。
    """
    from nptdms import TdmsFile
    with TdmsFile.open(file_path) as tdms_file:
        groups = tdms_file.groups()
        if not groups:
            return pd.DataFrame()  # 无组数据
        # 取第一个组
        group = groups[0]
        # 收集所有通道数据
        data_dict = {}
        for channel in group.channels():
            data_dict[channel.name] = channel[:]  # 获取通道数据
        return pd.DataFrame(data_dict)


def save_data(data, file_path):
    """
    将 DataFrame 保存为指定格式的文件。

    Args:
        data: pandas DataFrame 对象
        file_path: 保存文件路径

    Returns:
        str: 保存结果信息
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            data.to_csv(file_path, index=False, encoding='utf-8-sig')
            return f"CSV文件保存成功: {file_path}"
        elif ext in ('.xls', '.xlsx'):
            data.to_excel(file_path, index=False, sheet_name='Sheet1')
            return f"Excel文件保存成功: {file_path}"
        else:
            raise ValueError(f"不支持的保存格式: {ext}")
    except Exception as e:
        raise RuntimeError(f"保存文件失败: {str(e)}")