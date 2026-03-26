import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings; warnings.filterwarnings('ignore')

# 1. 数据加载
df = pd.read_csv(r'XGBoost_SHAP\datasets\mixed_output.csv')
X = df.drop(columns=['label'])
y = df['label']

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 类别加权
weight_per_class = y_train.value_counts().sort_index()
scale_pos = weight_per_class[0] / weight_per_class[1:]

# 4. 模型训练
clf = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    scale_pos_weight=scale_pos
)
clf.fit(X_train, y_train)

# 5. 模型评估
y_pred = clf.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print('测试集准确率:', clf.score(X_test, y_test))

# 6. SHAP解释
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# 7. SHAP图生成
shap.summary_plot(shap_values, X_test, feature_names=X.columns)