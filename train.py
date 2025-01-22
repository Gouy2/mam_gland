import numpy as np
from pathlib import Path
import pandas as pd
import os
import cv2

from module.load_data import load_cached_dataset, create_imgWithLabels
from module.load_data import process_images_for_patients,cache_dataset
from module.dataset import ImageDataset
from module.trainer import train


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

    # combined_images = np.concatenate([train_images, test_images], axis=0)

    #加载标签数据
    train_excel_path = './data/new_excel/chaoyang_retrospective_233.xlsx' 
    test_excel_path = './data/new_excel/chaoyang_prospective_190.xlsx'
    train_labels_df = pd.read_excel(train_excel_path)
    test_labels_df = pd.read_excel(test_excel_path)

    #创建图像标签对
    images_with_labels = create_imgWithLabels(train_images , train_labels_df , is_double=False, is_2cat=True)
    images_with_labels += create_imgWithLabels(test_images , test_labels_df , is_double=False, is_2cat=True)


    #创建数据集
    dataset = ImageDataset(images_with_labels)

    #超参数设置
    num_classes = 2
    input_channels = 2
    k_folds = 5
    batch_size = 16
    num_epochs = 80
    lr = 2e-3
    weight_decay = 1e-2

    train(dataset,num_classes,input_channels,k_folds,batch_size,num_epochs,lr,weight_decay)