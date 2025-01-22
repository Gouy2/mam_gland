import numpy as np
from pathlib import Path
import pandas as pd
import os
import cv2
import torch

from module.load_data import load_cached_dataset, create_imgWithLabels
from module.load_data import process_images_for_patients,cache_dataset
from module.dataset import ImageDataset
from module.trainer import train, test
from module.model import *

if __name__ == '__main__':

    # # 加载所有病人的图像
    # base_path = './data/new_huigu' 
    # base_path = './data/new_qianzhan' 
    # patient_images = process_images_for_patients(base_path, target_size=(224, 224), is_mask=False, is_double=False)
    

    # # # 缓存预处理后图像
    # cache_dataset(patient_images, f'cache/train_{len(patient_images)}_nonfo.npy', format='npy')

    #加载缓存图像数据
    # cache_path = 'cache/train_225_nonfo.npy'
    train_images = load_cached_dataset('cache/train_225_nonfo.npy', format='npy')
    test_images = load_cached_dataset('cache/train_180_nonfo.npy', format='npy')
    print("---加载图像数据完成---")

    # combined_images = np.concatenate([train_images, test_images], axis=0)

    #加载标签数据
    train_excel_path = './data/new_excel/chaoyang_retrospective_233.xlsx' 
    test_excel_path = './data/new_excel/chaoyang_prospective_190.xlsx'
    train_labels_df = pd.read_excel(train_excel_path)
    test_labels_df = pd.read_excel(test_excel_path)
    print("---加载标签数据完成---")

    #创建图像标签对
    images_with_labels = create_imgWithLabels(train_images , train_labels_df , is_double=False, is_2cat=True)
    images_with_labels += create_imgWithLabels(test_images , test_labels_df , is_double=False, is_2cat=True)

    
    #创建数据集
    dataset = ImageDataset(images_with_labels)
    # 抽出20%的数据作为测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=generator)

    #超参数设置
    num_classes = 2
    input_channels = 2
    k_folds = 5
    batch_size = 16
    num_epochs = 50
    lr = 1e-3
    weight_decay = 1e-2

    print("-------开始训练-------")
    print("使用设备：", torch.cuda.is_available())

    train(train_dataset,num_classes,input_channels,k_folds,batch_size,num_epochs,lr,weight_decay)

    print("-------开始测试-------")

    fold = 4
    epoch = 37
    day = 20250122
    time = 140513
    model_path = f'./results/{day}_{time}/models/best_model_fold_{fold}_epoch_{epoch}.pth'
    # test(test_dataset, model_path)

