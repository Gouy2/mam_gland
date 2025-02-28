import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class RadiomicsFeatureSelector:
    """影像组学特征选择与处理"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = None
        self.selection_method = None
        self.feature_selector = None
    
    def preprocess(self, df, numeric_only=True):
        """
        预处理特征DataFrame，填充缺失值，去除非数值列
        
        Args:
            df: 特征DataFrame
            numeric_only: 是否只保留数值列
            
        Returns:
            预处理后的DataFrame
        """
        # 复制以避免修改原始数据
        processed_df = df.copy()
        
        # 只保留数值列
        if numeric_only:
            processed_df = processed_df.select_dtypes(include=['number'])
        
        # 填充缺失值
        processed_df = processed_df.fillna(processed_df.mean())
        
        return processed_df
    
    def scale_features(self, df, fit=True):
        """
        标准化特征
        
        Args:
            df: 特征DataFrame
            fit: 是否拟合缩放器
            
        Returns:
            标准化后的特征数组
        """
        if fit:
            return self.scaler.fit_transform(df)
        else:
            return self.scaler.transform(df)
    
    def select_by_mi(self, X, y, k=20):
        """
        基于互信息选择特征
        
        Args:
            X: 特征数组
            y: 标签数组
            k: 选择的特征数量
            
        Returns:
            选择后的特征数组
        """
        self.selection_method = 'mutual_info'
        self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # 保存所选特征的索引
        self.selected_features = np.where(self.feature_selector.get_support())[0]
        
        return X_selected
    
    def select_by_rfe(self, X, y, n_features=20):
        """
        通过递归特征消除选择特征
        
        Args:
            X: 特征数组
            y: 标签数组
            n_features: 选择的特征数量
            
        Returns:
            选择后的特征数组
        """
        self.selection_method = 'rfe'
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # 保存所选特征的索引
        self.selected_features = np.where(self.feature_selector.get_support())[0]
        
        return X_selected
    
    def reduce_by_pca(self, X, n_components=0.95):
        """
        通过PCA降维
        
        Args:
            X: 特征数组
            n_components: 主成分数量或方差比例
            
        Returns:
            降维后的特征数组
        """
        self.selection_method = 'pca'
        self.feature_selector = PCA(n_components=n_components)
        X_reduced = self.feature_selector.fit_transform(X)
        
        return X_reduced
    
    def get_feature_names(self, original_features):
        """
        获取所选特征的名称
        
        Args:
            original_features: 原始特征名称列表
            
        Returns:
            所选特征名称列表
        """
        if self.selection_method == 'pca':
            return [f'PC{i+1}' for i in range(self.feature_selector.n_components_)]
        elif self.selected_features is not None:
            return [original_features[i] for i in self.selected_features]
        else:
            return original_features
    
    def plot_feature_importance(self, original_features, figsize=(12, 8)):
        """
        可视化特征重要性
        
        Args:
            original_features: 原始特征名称列表
            figsize: 图形大小
        """
        if self.selection_method == 'mutual_info':
            importance = self.feature_selector.scores_
            indices = np.argsort(importance)[-20:]  # 展示前20个特征
            features = [original_features[i] for i in indices]
            
            plt.figure(figsize=figsize)
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), features)
            plt.xlabel('互信息分数')
            plt.title('特征重要性 (基于互信息)')
            plt.tight_layout()
            plt.show()
            
        elif self.selection_method == 'rfe':
            # RFE只提供二元选择结果
            mask = self.feature_selector.get_support()
            features = [original_features[i] for i, selected in enumerate(mask) if selected]
            
            plt.figure(figsize=figsize)
            plt.barh(range(len(features)), np.ones(len(features)))
            plt.yticks(range(len(features)), features)
            plt.xlabel('已选择')
            plt.title('RFE所选特征')
            plt.tight_layout()
            plt.show()
            
        elif self.selection_method == 'pca':
            # 显示PCA解释方差比例
            plt.figure(figsize=figsize)
            plt.plot(np.cumsum(self.feature_selector.explained_variance_ratio_))
            plt.xlabel('主成分数量')
            plt.ylabel('解释方差累积比例')
            plt.title('PCA解释方差')
            plt.grid(True)
            plt.show()
            
            # 显示前两个主成分的特征贡献
            if len(original_features) > 0:
                plt.figure(figsize=figsize)
                components = pd.DataFrame(
                    self.feature_selector.components_[:2].T,
                    columns=['PC1', 'PC2'],
                    index=original_features
                )
                sns.heatmap(components, cmap='coolwarm', annot=True, fmt=".2f")
                plt.title('前两个主成分的特征贡献')
                plt.tight_layout()
                plt.show()
    
    def select_features_by_correlation(self, df, threshold=0.85):
        """
        根据相关性去除高度相关特征
        
        Args:
            df: 特征DataFrame
            threshold: 相关性阈值，高于此值的特征将被去除
            
        Returns:
            去除高相关后的DataFrame
        """
        # 计算相关性矩阵
        corr_matrix = df.corr().abs()
        
        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出高度相关的特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # 保存被选择的特征
        self.selected_features = [i for i, col in enumerate(df.columns) if col not in to_drop]
        self.selection_method = 'correlation'
        
        # 返回去除高相关特征后的DataFrame
        return df.drop(columns=to_drop)
    
    def select_stable_features(self, X, y, cv, method='mi', k=20, threshold=0.7):
        """
        选择在交叉验证中稳定的特征
        
        Args:
            X: 特征数组
            y: 标签数组
            cv: 交叉验证分割器
            method: 特征选择方法 ('mi'=互信息, 'rfe'=递归特征消除)
            k: 每次选择的特征数量
            threshold: 稳定性阈值，特征需要在多少比例的折中出现才保留
            
        Returns:
            稳定特征索引
        """
        feature_counts = np.zeros(X.shape[1])
        
        # 在每个折上选择特征
        for train_idx, _ in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            
            if method == 'mi':
                selector = SelectKBest(mutual_info_classif, k=k)
                selector.fit(X_train, y_train)
                selected = selector.get_support()
            elif method == 'rfe':
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                selector = RFE(estimator, n_features_to_select=k)
                selector.fit(X_train, y_train)
                selected = selector.get_support()
            else:
                raise ValueError(f"不支持的方法: {method}")
            
            # 累计特征在各折中出现的次数
            feature_counts += selected
        
        # 计算每个特征出现的比例
        feature_stability = feature_counts / cv.get_n_splits(X, y)
        
        # 选择出现频率超过阈值的特征
        stable_features = np.where(feature_stability >= threshold)[0]
        
        self.selected_features = stable_features
        self.selection_method = f'stable_{method}'
        
        return stable_features
