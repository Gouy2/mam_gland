import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor
import logging


class RadiomicsFeatureExtractor:
    """
    基于pyradiomics库的影像组学特征提取器
    """
    def __init__(self, config_path=None):
        """
        初始化特征提取器
        
        Args:
            config_path: 影像组学配置文件路径，如果为None，则使用默认配置
        """
        # 配置日志
        self.logger = logging.getLogger('radiomics')
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.ERROR)  # 设置为ERROR级别，减少输出
        
        # 初始化特征提取器
        if config_path and os.path.exists(config_path):
            # 使用UTF-8编码显式读取YAML文件
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            # 从字符串创建提取器而不是从文件
            self.extractor = featureextractor.RadiomicsFeatureExtractor(yamltext=yaml_content)
            print(f"使用自定义配置文件: {config_path}")
        else:
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
            print("使用pyradiomics默认配置")
        
        # 禁用形状特征，因为它们计算密集
        self.extractor.disableAllFeatures()
        self.extractor.enableFeaturesByName(firstorder=['Mean', 'Median', 'Energy', 'Entropy', 'Skewness', 'Kurtosis'])
        self.extractor.enableFeaturesByName(glcm=['Contrast', 'Correlation', 'JointEntropy'])
        self.extractor.enableFeaturesByName(glrlm=['GrayLevelNonUniformity', 'RunLengthNonUniformity'])
        
        # 打印激活的特征
        print(f"激活的特征类别: {self.extractor.enabledFeatures}")
        
    def extract_from_arrays(self, image_array, mask_array=None, patient_id=None):
        """
        从numpy数组提取影像组学特征
        
        Args:
            image_array: 3D或2D图像数组
            mask_array: 3D或2D掩码数组，如果为None，则使用全1掩码
            patient_id: 患者ID，用于标识
            
        Returns:
            特征字典
        """
        # 确保是numpy数组
        image_array = np.asarray(image_array)
        
        # 创建默认掩码
        if mask_array is None:
            mask_array = np.ones_like(image_array, dtype=np.uint8)
        else:
            mask_array = np.asarray(mask_array).astype(np.uint8)
        
        # 检查并修复维度
        if len(image_array.shape) == 2:
            image_array = image_array[np.newaxis, :, :]
            mask_array = mask_array[np.newaxis, :, :]
        
        # 转换为SimpleITK格式
        image = sitk.GetImageFromArray(image_array)
        mask = sitk.GetImageFromArray(mask_array)
        
        # 设置基本元数据
        image.SetSpacing((1.0, 1.0, 1.0))
        mask.SetSpacing((1.0, 1.0, 1.0))
        
        # 提取特征
        try:
            features = self.extractor.execute(image, mask)
            # 添加患者ID
            if patient_id is not None:
                features['PatientID'] = patient_id
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
            
    def extract_batch(self, image_arrays, mask_arrays=None, patient_ids=None):
        """
        批量提取特征
        
        Args:
            image_arrays: 图像数组列表
            mask_arrays: 掩码数组列表，可选
            patient_ids: 患者ID列表，可选
            
        Returns:
            特征DataFrame
        """
        all_features = []
        
        total = len(image_arrays)
        for i, image in enumerate(image_arrays):
            print(f"处理图像 {i+1}/{total}...")
            mask = None if mask_arrays is None else mask_arrays[i]
            patient_id = None if patient_ids is None else patient_ids[i]
            
            # 添加超时机制
            try:
                features = self.extract_from_arrays(image, mask, patient_id)
                if features:
                    print(f"图像 {i+1} 提取了 {len(features)} 个特征")
                    all_features.append(features)
                else:
                    print(f"图像 {i+1} 提取失败")
            except Exception as e:
                print(f"图像 {i+1} 提取时出错: {e}")
                continue
        
        # 转换为DataFrame
        if all_features:
            df = pd.DataFrame(all_features)
            # 删除类型列
            if 'diagnostics_Versions_PyRadiomics' in df.columns:
                diagnostics_columns = [col for col in df.columns if col.startswith('diagnostics_')]
                df = df.drop(diagnostics_columns, axis=1)
            return df
        else:
            return pd.DataFrame()