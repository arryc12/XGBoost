"""
数据 IO 处理模块，负责读取和保存不同格式的文件。
支持格式：.tdms, .csv, .xls, .xlsx
"""

import os
import pandas as pd
import numpy as np


def get_file_summary(file_path):
    """
    获取文件摘要信息。

    参数:
        file_path: 文件路径

    返回:
        文件摘要字符串
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.tdms':
            return _summary_tdms(file_path)
        elif ext == '.csv':
            return _summary_csv(file_path)
        elif ext in ('.xls', '.xlsx'):
            return _summary_excel(file_path)
        else:
            return f"不支持的文件格式: {ext}"
    except Exception as e:
        return f"读取失败: {str(e)}"


def _summary_tdms(file_path):
    """获取TDMS文件摘要"""
    try:
        from nptdms import TdmsFile
        with TdmsFile.open(file_path) as tdms_file:
            groups = tdms_file.groups()
            return f"TDMS文件，包含 {len(groups)} 个组"
    except ImportError:
        return "需要安装npTDMS库"


def _summary_csv(file_path):
    """获取CSV文件摘要"""
    df = pd.read_csv(file_path)
    return f"CSV文件，形状: {df.shape}"


def _summary_excel(file_path):
    """获取Excel文件摘要"""
    df = pd.read_excel(file_path, sheet_name=0)
    return f"Excel文件，形状: {df.shape}"


def load_data(file_path):
    """
    读取文件并返回完整的 pandas DataFrame。
    对于 TDMS 文件，尝试将第一个组的所有通道合并为 DataFrame。

    参数:
        file_path: 文件路径

    返回:
        pandas DataFrame 对象
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.tdms':
            return _load_tdms(file_path)
        elif ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ('.xls', '.xlsx'):
            return pd.read_excel(file_path, sheet_name=0)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    except Exception as e:
        raise RuntimeError(f"加载文件失败: {str(e)}")


def _load_tdms(file_path):
    """将 TDMS 文件转换为 DataFrame"""
    from nptdms import TdmsFile
    with TdmsFile.open(file_path) as tdms_file:
        groups = tdms_file.groups()
        if not groups:
            return pd.DataFrame()
        group = groups[0]
        data_dict = {}
        for channel in group.channels():
            data_dict[channel.name] = channel[:]
        return pd.DataFrame(data_dict)


def save_data(data, file_path):
    """
    将 DataFrame 保存为指定格式的文件。

    参数:
        data: pandas DataFrame 对象
        file_path: 保存文件路径

    返回:
        保存结果信息
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
