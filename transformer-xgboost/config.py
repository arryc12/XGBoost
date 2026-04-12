#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "erect" / "data0"
OUTPUT_DIR = PROJECT_ROOT / "transformer-xgboost" / "outputs"

# 数据参数
CHUNK_SIZE = 2000
INPUT_COLS = [4, 5]  # 进口压力, 出口压力

# Transformer参数
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
SEQ_LEN = CHUNK_SIZE

# XGBoost参数
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

# 训练参数
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# 优化配置
OPTIMIZE_HYPERPARAMS = True
SSA_N_SPARROW = 20
SSA_N_ITER = 30

# 流型标签
FLOW_REGIMES = {
    "annular_flow": 0,
    "bubble_flow": 1,
    "bullet_flow": 2,
    "slug_flow": 3,
}