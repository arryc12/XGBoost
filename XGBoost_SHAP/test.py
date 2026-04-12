
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import warnings; warnings.filterwarnings('ignore')


class SSA:
    """麻雀搜索算法 (Sparrow Search Algorithm) 用于超参数优化
    
    模拟麻雀的觅食行为，将麻雀群体分为三类：
    - 发现者（探索者）：负责寻找食物，具有较好的适应度值
    - 跟随者（追随者）：跟随发现者觅食
    - 侦察者：负责警戒，发现危险时随机移动
    """
    
    def __init__(self, n_sparrow=30, n_iter=50, bounds=None, 
                 discoverer_ratio=0.2, scout_ratio=0.2, verbose=True):
        """
        初始化SSA算法
        
        参数:
            n_sparrow: 麻雀数量
            n_iter: 迭代次数
            bounds: 参数边界 [(min, max), ...]
            discoverer_ratio: 发现者比例（默认0.2）
            scout_ratio: 侦察者比例（默认0.2）
            verbose: 是否打印优化过程
        """
        self.n_sparrow = n_sparrow
        self.n_iter = n_iter
        self.bounds = bounds
        self.discoverer_ratio = discoverer_ratio
        self.scout_ratio = scout_ratio
        self.verbose = verbose
        self.best_params = None
        self.best_score = -np.inf
        self.history = []
        
        self.n_discoverer = max(1, int(n_sparrow * discoverer_ratio))
        self.n_scout = max(1, int(n_sparrow * scout_ratio))
    
    def _init_sparrows(self):
        """初始化麻雀群：在参数边界内随机生成初始位置"""
        n_dims = len(self.bounds)
        sparrows = np.zeros((self.n_sparrow, n_dims))
        for i in range(n_dims):
            low, high = self.bounds[i]
            sparrows[:, i] = np.random.uniform(low, high, self.n_sparrow)
        return sparrows
    
    def _get_params_from_position(self, position):
        """将位置向量转换为参数字典"""
        params = {}
        param_names = ['n_estimators', 'max_depth', 'learning_rate', 
                       'subsample', 'colsample_bytree', 'min_child_weight',
                       'gamma', 'reg_alpha', 'reg_lambda']
        int_params = ['n_estimators', 'max_depth', 'min_child_weight']
        for i, name in enumerate(param_names):
            if i < len(position):
                if name in int_params:
                    params[name] = int(round(position[i]))
                else:
                    params[name] = position[i]
        return params
    
    def _objective(self, position, X_train, y_train):
        """目标函数：使用交叉验证评估模型"""
        params = self._get_params_from_position(position)
        params['random_state'] = 42
        params['use_label_encoder'] = False
        params['eval_metric'] = 'logloss'
        
        model = XGBClassifier(**params)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            return scores.mean()
        except:
            return -999
    
    def _evaluate_all(self, sparrows, X_train, y_train):
        """评估所有麻雀的适应度值"""
        fitness = np.zeros(self.n_sparrow)
        for i in range(self.n_sparrow):
            fitness[i] = self._objective(sparrows[i], X_train, y_train)
        return fitness
    
    def fit(self, X_train, y_train):
        """运行SSA优化算法"""
        n_dims = len(self.bounds)
        sparrows = self._init_sparrows()
        fitness = self._evaluate_all(sparrows, X_train, y_train)
        
        best_idx = np.argmax(fitness)
        best_pos = sparrows[best_idx].copy()
        best_fit = fitness[best_idx]
        
        worst_idx = np.argmin(fitness)
        worst_pos = sparrows[worst_idx].copy()
        
        for iteration in range(self.n_iter):
            sorted_idx = np.argsort(fitness)[::-1]
            sparrows_sorted = sparrows[sorted_idx]
            fitness_sorted = fitness[sorted_idx]
            
            discoverer_pos = sparrows_sorted[:self.n_discoverer].copy()
            follower_pos = sparrows_sorted[self.n_discoverer:-self.n_scout].copy()
            scout_pos = sparrows_sorted[-self.n_scout:].copy()
            
            r2 = np.random.uniform(0, 1)
            for i in range(self.n_discoverer):
                if r2 < 0.8:
                    discoverer_pos[i] = discoverer_pos[i] * np.exp(-i / (np.random.uniform(0.5, 1) * self.n_iter))
                else:
                    discoverer_pos[i] = discoverer_pos[i] + np.random.normal() * 10
                for d in range(n_dims):
                    low, high = self.bounds[d]
                    discoverer_pos[i, d] = np.clip(discoverer_pos[i, d], low, high)
            
            for i in range(len(follower_pos)):
                actual_idx = i + self.n_discoverer
                if actual_idx > self.n_sparrow / 2:
                    follower_pos[i] = np.random.normal() * np.exp((best_pos - follower_pos[i]) / (actual_idx + 1) ** 2)
                else:
                    A_plus = (follower_pos[i] * np.ones(n_dims)).T @ np.linalg.pinv(np.ones((1, n_dims)))
                    follower_pos[i] = best_pos + np.abs(follower_pos[i] - best_pos) * (2 * np.random.uniform(0, 1) - 1)
                for d in range(n_dims):
                    low, high = self.bounds[d]
                    follower_pos[i, d] = np.clip(follower_pos[i, d], low, high)
            
            r1 = np.random.uniform(0, 1)
            for i in range(self.n_scout):
                if r1 < 0.8:
                    scout_pos[i] = best_pos + np.random.normal() * np.abs(scout_pos[i] - best_pos)
                else:
                    scout_pos[i] = scout_pos[i] + (2 * np.random.uniform(0, 1) - 1) * np.abs(scout_pos[i] - worst_pos) / (fitness_sorted[-(i+1)] - np.min(fitness_sorted) + 1e-8)
                for d in range(n_dims):
                    low, high = self.bounds[d]
                    scout_pos[i, d] = np.clip(scout_pos[i, d], low, high)
            
            sparrows = np.vstack([discoverer_pos, follower_pos, scout_pos])
            fitness = self._evaluate_all(sparrows, X_train, y_train)
            
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fit:
                best_fit = fitness[current_best_idx]
                best_pos = sparrows[current_best_idx].copy()
            
            worst_idx = np.argmin(fitness)
            worst_pos = sparrows[worst_idx].copy()
            
            self.history.append(best_fit)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}/{self.n_iter}, 最优准确率: {best_fit:.4f}")
        
        self.best_params = self._get_params_from_position(best_pos)
        self.best_score = best_fit
        return self


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


# 4. SSA超参数优化
print("=" * 50)
print("使用SSA算法自动优化XGBoost参数")
print("=" * 50)

param_bounds = [
    (100, 500),      # 树的数量
    (3, 15),         # 树的最大深度
    (0.01, 0.2),     # 学习率
    (0.6, 1.0),      # 样本采样比例
    (0.6, 1.0),      # 特征采样比例
    (1, 10),         # 子节点最小权重
    (0, 1),          # 分裂最小损失增益
    (0, 1),          # L1正则化系数
    (0, 1),          # L2正则化系数
]

ssa = SSA(n_sparrow=20, n_iter=30, bounds=param_bounds, verbose=True)
ssa.fit(X_train, y_train)

print("\n最优参数:")
for k, v in ssa.best_params.items():
    print(f"  {k}: {v:.4f}")
print(f"交叉验证准确率: {ssa.best_score:.4f}")

# 5. 使用优化参数训练模型
best_params = ssa.best_params.copy()
best_params['scale_pos_weight'] = scale_pos
best_params['random_state'] = 42
best_params['use_label_encoder'] = False
best_params['eval_metric'] = 'logloss'

clf = XGBClassifier(**best_params)
clf.fit(X_train, y_train)

# 6. 模型评估
y_pred = clf.predict(X_test)
print("\n" + "=" * 50)
print("模型评估结果")
print("=" * 50)
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 7. 优化过程可视化
plt.figure(figsize=(10, 4))
plt.plot(ssa.history, 'b-', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('SSA优化过程')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'XGBoost_SHAP\ssa_optimization.png', dpi=150)
plt.show()

# 8. SHAP解释
print("\n" + "=" * 50)
print("SHAP特征重要性分析")
print("=" * 50)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# 9. SHAP图生成
shap.summary_plot(shap_values, X_test, feature_names=X.columns)