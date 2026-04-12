#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器用于时序特征提取"""
    
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = d_model
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入 tensor, 形状为 (batch_size, seq_len, input_dim)
        
        返回:
            特征 tensor, 形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 全局平均池化 + CLS token 策略
        # 使用平均池化
        x = x.mean(dim=1)
        
        # 输出映射
        x = self.output_layer(x)
        
        return x


class ConvolutionalTransformer(nn.Module):
    """带卷积预处理的Transformer编码器"""
    
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        conv_channels: list = None
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = d_model
        if conv_channels is None:
            conv_channels = [32, 64]
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 1D卷积层
        conv_layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 投影到d_model维度
        self.projection = nn.Linear(conv_channels[-1], d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入 tensor, 形状为 (batch_size, seq_len, input_dim)
        
        返回:
            特征 tensor, 形状为 (batch_size, output_dim)
        """
        # 转换为 (batch, channels, seq) 用于卷积
        x = x.transpose(1, 2)
        
        # 卷积
        x = self.conv_layers(x)
        
        # 转换回 (batch, seq, channels)
        x = x.transpose(1, 2)
        
        # 投影
        x = self.projection(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 输出
        x = self.output_layer(x)
        
        return x


def get_transformer_model(
    model_type: str = "basic",
    input_dim: int = 2,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    output_dim: int = 64
) -> nn.Module:
    """
    获取Transformer模型
    
    参数:
        model_type: "basic" 或 "conv"
        input_dim: 输入维度（通道数）
        d_model: 模型维度
        nhead: 注意力头数
        num_layers: 编码器层数
        dim_feedforward: 前馈网络维度
        dropout: Dropout比例
        output_dim: 输出特征维度
    
    返回:
        Transformer模型
    """
    if model_type == "conv":
        model = ConvolutionalTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim
        )
    else:
        model = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim
        )
    
    return model


if __name__ == "__main__":
    print("测试Transformer模型...")
    
    # 创建模型
    model = get_transformer_model(
        model_type="basic",
        input_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        output_dim=64
    )
    
    # 测试输入
    batch_size = 4
    seq_len = 2000
    input_dim = 2
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")