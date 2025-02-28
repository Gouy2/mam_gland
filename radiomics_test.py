import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib

from custom_radiomics.feature_extractor import RadiomicsFeatureExtractor
from custom_radiomics.feature_selector import RadiomicsFeatureSelector
from utils.load_data import load_cached_dataset, create_imgWithLabels
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from scipy.stats import uniform, randint

def extract_radiomics_features(images, config_path='radiomics_config.yaml'):
    """从图像中提取影像组学特征"""
    print("开始提取影像组学特征...")
    extractor = RadiomicsFeatureExtractor(config_path)
    features_df = extractor.extract_batch(images)
    print(f"提取完成，共获取 {features_df.shape[1]} 个特征")
    return features_df

def prepare_radiomics_data(features_df, labels_df, patient_id_column='PatientID', label_column='label'):
    """准备影像组学特征数据集"""
    # 合并特征和标签
    if patient_id_column in features_df.columns and patient_id_column in labels_df.columns:
        merged_df = pd.merge(features_df, labels_df, on=patient_id_column)
    else:
        # 如果没有患者ID，假设顺序相同
        merged_df = pd.concat([features_df, labels_df], axis=1)
    
    # 获取标签列
    for col in labels_df.columns:
        if col != patient_id_column:
            label_column = col
            break
    
    # 分离特征和标签
    feature_columns = [col for col in features_df.columns if col != patient_id_column]
    X = merged_df[feature_columns].values
    
    # 将标签转换为连续整数（从0开始）
    y_raw = merged_df[label_column].values
    unique_labels = np.unique(y_raw)
    print(f"原始标签数量: {len(unique_labels)}")
    
    # 创建标签映射（确保从0开始的连续整数）
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y_raw])
    
    print(f"重编码后标签数量: {len(np.unique(y))}")
    
    return X, y, feature_columns

def evaluate_radiomics_models(X, y, feature_names, output_dir='./results/radiomics'):
    """使用SVM评估影像组学特征"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化特征选择器
    selector = RadiomicsFeatureSelector()
    
    # 预处理特征
    X_scaled_robust = RobustScaler().fit_transform(pd.DataFrame(X, columns=feature_names))
    X_scaled_minmax = MinMaxScaler().fit_transform(pd.DataFrame(X, columns=feature_names))
    
    # 设置交叉验证
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 添加更多特征选择方法
    selection_methods = {
        'NoSelection': lambda X, y: (X, feature_names),
        'MutualInfo': lambda X, y: (selector.select_by_mi(X, y, k=min(20, X.shape[1])), 
                                  selector.get_feature_names(feature_names)),
        'PCA': lambda X, y: (selector.reduce_by_pca(X, n_components=min(10, X.shape[1]//2)), 
                           [f'PC{i+1}' for i in range(min(10, X.shape[1]//2))]),
        'RF_Selection': lambda X, y: (SelectFromModel(RandomForestClassifier(n_estimators=100)).fit_transform(X, y),
                                   [feature_names[i] for i in SelectFromModel(
                                       RandomForestClassifier(n_estimators=100)
                                   ).fit(X, y).get_support(indices=True)])
    }
    
    # 结果存储
    results = []
    
    # 对每种特征选择方法进行评估
    for selection_name, selection_func in selection_methods.items():
        print(f"使用特征选择方法: {selection_name}")
        # 应用特征选择
        X_selected, selected_features = selection_func(X_scaled_robust, y)
        
        # 创建包含SVM的Pipeline
        classifier = Pipeline([
            ('scaler', StandardScaler()),  # 可选的额外标准化
            ('classifier', SVC(probability=True))
        ])

        # 保持现有的参数分布不变
        param_distributions = {
            'classifier__C': uniform(0.1, 100),
            'classifier__gamma': uniform(0.001, 1.0),
            'classifier__kernel': ['rbf', 'linear', 'poly'],
            'classifier__degree': randint(2, 5),
            'classifier__coef0': uniform(0.0, 10.0)
        }

        random_search = RandomizedSearchCV(
            classifier, param_distributions, n_iter=50, cv=cv, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        # 训练模型
        print("训练SVM模型...")
        random_search.fit(X_selected, y)
        best_model = random_search.best_estimator_
        
        # 保存最佳模型
        model_filename = os.path.join(output_dir, f'{selection_name}_SVM_model.pkl')
        joblib.dump(best_model, model_filename)
        
        # 使用交叉验证评估
        fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        
        for train_idx, test_idx in cv.split(X_selected, y):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练模型
            best_model.fit(X_train, y_train)
            
            # 预测类别和概率
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类概率
            
            # 计算指标
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(precision_score(y_test, y_pred, average='macro'))
            fold_metrics['recall'].append(recall_score(y_test, y_pred, average='macro'))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, average='macro'))
            
            # 计算AUC（只适用于二分类）
            try:
                fold_metrics['auc'].append(roc_auc_score(y_test, y_prob))
            except:
                fold_metrics['auc'].append(0)  # 如果出错（例如，只有一个类别），则设为0
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        
        # 保存结果
        result = {
            'FeatureSelection': selection_name,
            'NumFeatures': X_selected.shape[1],
            'Accuracy': avg_metrics['accuracy'],
            'Precision': avg_metrics['precision'],
            'Recall': avg_metrics['recall'],
            'F1': avg_metrics['f1'],
            'AUC': avg_metrics['auc'],  # 添加AUC指标
            'BestParams': random_search.best_params_
        }
        results.append(result)
        
        # 绘制ROC曲线（对于二分类）
        if len(np.unique(y)) == 2:
            best_model.fit(X_selected, y)
            y_prob = best_model.predict_proba(X_selected)[:, 1]
            
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC曲线 (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title(f'ROC曲线: {selection_name}_SVM')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(output_dir, f'{selection_name}_SVM_roc.png'))
            plt.close()
        
        # 绘制混淆矩阵
        best_model.fit(X_selected, y)
        y_pred = best_model.predict(X_selected)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'混淆矩阵: {selection_name}_SVM')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.savefig(os.path.join(output_dir, f'{selection_name}_SVM_cm.png'))
        plt.close()
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'svm_results.csv'), index=False)
    
    # 打印结果
    print("\n特征选择方法比较:")
    for i, row in results_df.iterrows():
        print(f"{row['FeatureSelection']}: 准确率={row['Accuracy']:.4f}, F1={row['F1']:.4f}, AUC={row['AUC']:.4f}")
    
    return results_df

def main():
    """影像组学特征提取与评估主函数"""
    # 创建输出目录
    output_dir = './results/radiomics'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载图像数据
    print("加载图像数据...")
    train_images = load_cached_dataset('cache/train_225_nonfo.npy', format='npy')
    
    # 2. 加载标签数据
    print("加载标签数据...")
    train_excel_path = './data/new_excel/chaoyang_retrospective_233.xlsx'
    train_labels_df = pd.read_excel(train_excel_path)
    
    # 使用已有函数处理图像和标签
    images_with_labels = create_imgWithLabels(train_images, train_labels_df, is_double=False, is_2cat=True)
    
    # 提取处理后的图像和标签
    processed_images = [item[0] for item in images_with_labels]
    processed_labels = [item[1] for item in images_with_labels]
    
    # 创建带有处理后标签的DataFrame
    processed_labels_df = pd.DataFrame({
        'PatientID': range(len(processed_labels)),  # 创建ID列
        'N分期': processed_labels  # 使用处理后的标签
    })
    
    # 3. 提取影像组学特征
    train_features_df = extract_radiomics_features(processed_images, config_path='radiomics_config.yaml')
    
    # 4. 准备数据
    X, y, feature_names = prepare_radiomics_data(train_features_df, processed_labels_df, 
                                               patient_id_column='PatientID', label_column='N分期')
    
    # 5. 评估模型
    results = evaluate_radiomics_models(X, y, feature_names, output_dir)
    
    # 6. 保存特征
    train_features_df.to_csv(os.path.join(output_dir, 'radiomics_features.csv'), index=False)
    
    # 7. 输出最佳模型
    if not results.empty:
        best_result = results.loc[results['F1'].idxmax()]
        print(f"\n最佳模型: {best_result['FeatureSelection']}, F1: {best_result['F1']:.4f}")

if __name__ == "__main__":
    main() 