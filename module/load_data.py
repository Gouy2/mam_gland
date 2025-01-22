from pathlib import Path
import pydicom
import nibabel as nib
import cv2
import numpy as np
from skimage import exposure
import pandas as pd
import os

def load_img(img_path, format = None):

    if format == 'dcm':
        dicom_data = pydicom.dcmread(img_path)
        image = dicom_data.pixel_array.astype(np.float32)
    elif format == 'jpg':
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {format}")
        
    # 归一化
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  

    # 适应性直方图均衡化
    image = exposure.equalize_adapthist(image, clip_limit=0.05)

    # 转换尺寸
    # dcm_image_resized = cv2.resize(dcm_image, target_size)

    return image

def load_dcm(dcm_path):
    # 加载 DICOM 图像
    dicom_data = pydicom.dcmread(dcm_path)
    dcm_image = dicom_data.pixel_array.astype(np.float32)

    dcm_image = (dcm_image - np.min(dcm_image)) / (np.max(dcm_image) - np.min(dcm_image))  

    dcm_image = exposure.equalize_adapthist(dcm_image, clip_limit=0.05)

    # 转换尺寸（512x512）
    # dcm_image_resized = cv2.resize(dcm_image, target_size)

    return dcm_image

def load_jpg(jpg_path, target_size=(224, 224)):
    # 加载 DICOM 图像
    jpg_image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    jpg_image = jpg_image.astype(np.float32)

    jpg_image_normalized = (jpg_image - np.min(jpg_image)) / (np.max(jpg_image) - np.min(jpg_image))  

    # 转换尺寸（512x512）
    jpg_image_resized = cv2.resize(jpg_image_normalized, target_size)

    return jpg_image_resized


def load_nii(nii_path, focus_strength=5):
    # 加载 NIfTI 图像
    nii_data = nib.load(nii_path)
    nii_image = nii_data.get_fdata()

    # 去掉第三维度
    nii_image = np.squeeze(nii_image)

    # 转置 NIfTI 图像，使其与 DICOM 图像对齐
    nii_image_transposed = np.transpose(nii_image, (1, 0))  # 转置为 (2294, 1914)

    mask = np.where(nii_image_transposed > 0, 1, 0).astype(np.uint8)

    # 找到包含所有肿瘤区域的最小矩形边界
    rows = np.any(mask, axis=1)  # 找出每一行是否有肿瘤区域
    cols = np.any(mask, axis=0)  # 找出每一列是否有肿瘤区域

    # 找到肿瘤区域的上下左右边界
    top = np.argmax(rows)         # 最上面一行
    bottom = len(rows) - np.argmax(rows[::-1])  # 最下面一行
    left = np.argmax(cols)        # 最左面一列
    right = len(cols) - np.argmax(cols[::-1])  # 最右面一列

    # 计算原始矩形的宽度和高度
    height = bottom - top
    width = right - left

    # 选择正方形的边长，取宽和高的最大值
    side_length = max(height, width)

    # 初始化扩展量
    top_extend = bottom_extend = left_extend = right_extend = 0

    # 计算上下需要扩展的距离
    if side_length - height > 0:  # 如果高度小于正方形边长
        total_vertical_extend = side_length - height
        top_extend = min(total_vertical_extend // 2, top)  # 尝试向上扩展
        bottom_extend = total_vertical_extend - top_extend  # 剩余的扩展量放在下边

        # 检查下边扩展是否超出图像边界
        if bottom_extend > (len(mask) - bottom):  # 下边扩展超出边界
            bottom_extend = len(mask) - bottom  # 调整下边扩展
            top_extend = total_vertical_extend - bottom_extend  # 调整上边扩展

    # 计算左右需要扩展的距离
    if side_length - width > 0:  # 如果宽度小于正方形边长
        total_horizontal_extend = side_length - width
        left_extend = min(total_horizontal_extend // 2, left)  # 尝试向左扩展
        right_extend = total_horizontal_extend - left_extend  # 剩余的扩展量放在右边

        # 检查右边扩展是否超出图像边界
        if right_extend > (len(mask[0]) - right):  # 右边扩展超出边界
            right_extend = len(mask[0]) - right  # 调整右边扩展
            left_extend = total_horizontal_extend - right_extend  # 调整左边扩展


    # 扩展矩形区域以获得正方形区域
    # 使用 padding 来扩展掩模矩阵
    mask = np.pad(mask[top:bottom, left:right], 
                    ((top_extend, bottom_extend), (left_extend, right_extend)), 
                    mode='constant', constant_values=0)

    # # 调整掩码大小
    # mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 计算正方形左上角和右下角的坐标
    top_left = (top - top_extend, left - left_extend)
    bottom_right = (bottom + bottom_extend, right + right_extend)

    # np.savetxt('mask.txt', mask)

    # output_file = 'tumor_region_info.txt'

    # # 将变量格式化为字符串
    # region_info = f"""
    # Side Length: {side_length}
    # """

    # Top: {top}
    # Bottom: {bottom}
    # Left: {left}
    # Right: {right}
    # Height: {height}
    # Width: {width}

    # 将信息写入文本文件
    # with open(output_file, 'a') as f:
    #     f.write(region_info)
    #     f.write('\n')

    # 使用距离变换
    dist_transform = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    dist_transform_normalized = cv2.normalize(dist_transform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 将标记区域设置为 1，其余区域根据距离逐渐降低
    mask = np.power(1 - dist_transform_normalized, focus_strength)
    
    return mask , top_left, bottom_right

def load_mask(nii_path, focus_strength=5):  # 无距离变换
    # 加载 NIfTI 图像
    nii_data = nib.load(nii_path)
    nii_image = nii_data.get_fdata()

    # 去掉第三维度
    nii_image = np.squeeze(nii_image)

    # 转置 NIfTI 图像，使其与 DICOM 图像对齐
    nii_image_transposed = np.transpose(nii_image, (1, 0))  # 转置为 (2294, 1914)

    mask = np.where(nii_image_transposed > 0, 1, 0).astype(np.uint8)

    # 找到包含所有肿瘤区域的最小矩形边界
    rows = np.any(mask, axis=1)  # 找出每一行是否有肿瘤区域
    cols = np.any(mask, axis=0)  # 找出每一列是否有肿瘤区域

    # 找到肿瘤区域的上下左右边界
    top = np.argmax(rows)         # 最上面一行
    bottom = len(rows) - np.argmax(rows[::-1])  # 最下面一行
    left = np.argmax(cols)        # 最左面一列
    right = len(cols) - np.argmax(cols[::-1])  # 最右面一列

    # 计算原始矩形的宽度和高度
    height = bottom - top
    width = right - left

    # 选择正方形的边长，取宽和高的最大值
    side_length = max(height, width)

    # 初始化扩展量
    top_extend = bottom_extend = left_extend = right_extend = 0

    # 计算上下需要扩展的距离
    if side_length - height > 0:  # 如果高度小于正方形边长
        total_vertical_extend = side_length - height
        top_extend = min(total_vertical_extend // 2, top)  # 尝试向上扩展
        bottom_extend = total_vertical_extend - top_extend  # 剩余的扩展量放在下边

        # 检查下边扩展是否超出图像边界
        if bottom_extend > (len(mask) - bottom):  # 下边扩展超出边界
            bottom_extend = len(mask) - bottom  # 调整下边扩展
            top_extend = total_vertical_extend - bottom_extend  # 调整上边扩展

    # 计算左右需要扩展的距离
    if side_length - width > 0:  # 如果宽度小于正方形边长
        total_horizontal_extend = side_length - width
        left_extend = min(total_horizontal_extend // 2, left)  # 尝试向左扩展
        right_extend = total_horizontal_extend - left_extend  # 剩余的扩展量放在右边

        # 检查右边扩展是否超出图像边界
        if right_extend > (len(mask[0]) - right):  # 右边扩展超出边界
            right_extend = len(mask[0]) - right  # 调整右边扩展
            left_extend = total_horizontal_extend - right_extend  # 调整左边扩展


    # 扩展矩形区域以获得正方形区域
    # 使用 padding 来扩展掩模矩阵
    mask = np.pad(mask[top:bottom, left:right], 
                    ((top_extend, bottom_extend), (left_extend, right_extend)), 
                    mode='constant', constant_values=0)

    # # 调整掩码大小
    # mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 计算正方形左上角和右下角的坐标
    top_left = (top - top_extend, left - left_extend)
    bottom_right = (bottom + bottom_extend, right + right_extend)

    # np.savetxt('mask.txt', mask)

    # output_file = 'tumor_region_info.txt'

    # # 将变量格式化为字符串
    # region_info = f"""
    # Side Length: {side_length}
    # """

    # Top: {top}
    # Bottom: {bottom}
    # Left: {left}
    # Right: {right}
    # Height: {height}
    # Width: {width}

    # 将信息写入文本文件
    # with open(output_file, 'a') as f:
    #     f.write(region_info)
    #     f.write('\n')

    # 使用距离变换
    # dist_transform = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    # dist_transform_normalized = cv2.normalize(dist_transform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # # 将标记区域设置为 1，其余区域根据距离逐渐降低
    # mask = np.power(1 - dist_transform_normalized, focus_strength)
    
    return mask , top_left, bottom_right

def cache_dataset(data, cache_path, format='npy'):
    """
    缓存数据集到指定路径
    
    参数:
    data: 要缓存的数据
    cache_path: 缓存文件路径
    format: 文件格式 ('npy', 'h5', 'pkl', 'joblib')
    """
    cache_path = Path(cache_path)
    
    if format == 'npy':
        np.save(cache_path, np.array(data))
    else:
        raise ValueError(f"Unsupported format: {format}")
    
def load_cached_dataset(cache_path, format='npy'):
    """
    加载缓存的数据集
    
    参数:
    cache_path: 缓存文件路径
    format: 文件格式 ('npy', 'h5', 'pkl', 'joblib')
    """
    cache_path = Path(cache_path)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    if format == 'npy':
        return np.load(cache_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
def create_imgWithLabels(patient_images , labels_df , is_double = False , is_2cat = True):

    # 补全标签并构建 images_with_labels 列表
    images_with_labels = []
    labels = []

    if is_double:
        # 确保图像数量是标签数量的两倍
        assert len(patient_images) == 2 * len(labels_df), "图像数量应该是标签数量的两倍"
        
        for i in range(0, len(patient_images), 2):  # 每次迭代处理两张图像
            label_index = i // 2  # 获取对应的标签索引
            label = labels_df.iloc[label_index]['N分期']  # 获取标签
            
            # 如果标签为 NaN，则用均值填充
            if pd.isna(label):
                label = 1.0

            if is_2cat:   # 二分类       
                if label == 2.0 or label == 3.0:
                    label = 1.0
            
            # 将两张连续的图像都与同一个标签配对
            images_with_labels.append((patient_images[i], label))
            images_with_labels.append((patient_images[i+1], label))
            labels.append(label)
            labels.append(label)

    else:
        for i, patient_input in enumerate(patient_images):
            label = labels_df.iloc[i]['N分期']  # 按顺序获取对应的标签

            # 如果标签为 NaN，则用均值填充
            if pd.isna(label):
                label = 1.0

            if is_2cat:   # 二分类       
                if label == 2.0 or label == 3.0:
                    label = 1.0
            
            images_with_labels.append((patient_input, label))
            labels.append(label)

    return images_with_labels

def create_imgWithLabels_2(patient_images, labels_df):
    # 补全标签并构建 images_with_labels 列表
    images_with_labels = []
    labels = []
    
    # 确保图像数量是标签数量的两倍
    assert len(patient_images) == 2 * len(labels_df), "图像数量应该是标签数量的两倍"
    
    for i in range(0, len(patient_images), 2):  # 每次迭代处理两张图像
        label_index = i // 2  # 获取对应的标签索引
        label = labels_df.iloc[label_index]['N分期']  # 获取标签
        
        # 如果标签为 NaN，则用均值填充
        if pd.isna(label):
            label = 1.0
        elif label == 2.0 or label == 3.0:
            label = 1.0
        
        # 将两张连续的图像都与同一个标签配对
        images_with_labels.append((patient_images[i], label))
        images_with_labels.append((patient_images[i+1], label))
        labels.append(label)
        labels.append(label)

    return images_with_labels

def get_labels(excel_path):
    # 补全标签并构建 images_with_labels 列表
    labels_df = pd.read_excel(excel_path)

    labels = []
    
    # 确保图像数量是标签数量的两倍
    assert 466 == 2 * len(labels_df), "图像数量应该是标签数量的两倍"
    
    for i in range(0, 466, 2):  # 每次迭代处理两张图像
        label_index = i // 2  # 获取对应的标签索引
        label = labels_df.iloc[label_index]['N分期']  # 获取标签
        
        # 如果标签为 NaN，则用均值填充
        if pd.isna(label):
            label = 1.0
        elif label == 2.0 or label == 3.0:
            label = 1.0
        
        labels.append(label)
        labels.append(label)

    return labels

def process_images_for_patients(base_path, target_size=(224, 224), is_mask=True, is_double=False, count = 0):

    # 获取所有病人文件夹
    all_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

    # 按病人分组文件夹，每两个文件夹为一个病人
    patient_folders = []
    for i in range(0, len(all_folders), 2):
        
        if i + 1 < len(all_folders):
            patient_folders.append([all_folders[i], all_folders[i + 1]])
        
    # 加载每个文件夹中的图像
    patient_images = []

    for folder_pair in patient_folders:

        all_images = []
        for folder in folder_pair:
            count += 1

            dcm_file = None
            nii_file = None
            jpg_file = None

            # 查找对应的 .dcm 和 .nii 文件
            for file in os.listdir(os.path.join(base_path, folder)):
                if file.endswith('.dcm'):
                    dcm_file = os.path.join(base_path, folder, file)
                elif file.endswith('.nii.gz'):
                    nii_file = os.path.join(base_path, folder, file)
                elif file.endswith('.jpg'):
                    jpg_file = os.path.join(base_path, folder, file)                               

            if dcm_file and nii_file:
                # 读取并处理 .dcm 和 .nii 图像
                dcm_image = load_img(dcm_file,'dcm')
                nii_mask , top_left, bottom_right = load_nii(nii_file)
                image = dcm_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

                if is_mask:
                    image = image * nii_mask

                image = cv2.resize(image, target_size)
                            
                # npimage = np.array(focused_image)
                # np.savetxt(npimage,f'./txt/{count}.txt')
                # print(npimage)
                # np.savetxt(f"./txt/{count}.txt", npimage, fmt="%.1f", delimiter=",")

                # import matplotlib.pyplot as plt
                # 保存图片
                # plt.imshow(dcm_image, cmap='gray')
                # # plt.imshow(focused_image, cmap='gray')
                # plt.savefig(f"./image/cur{count}.png")
                
                all_images.append(image)
                # patient_images.append(dcm_image)

            elif jpg_file and nii_file:
                # 读取并处理 .jpg 和 .nii 图像
                jpg_image = load_img(jpg_file,'jpg')
                nii_mask , top_left, bottom_right = load_nii(nii_file)
                image = jpg_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

                if is_mask:
                    image = image * nii_mask

                image = cv2.resize(image, target_size)
                
                all_images.append(image)
                # patient_images.append(jpg_image)

            # 确实nii文件
            elif dcm_file and not nii_file:

                dcm_image = load_img(dcm_file,'dcm')
                dcm_image = cv2.resize(dcm_image, target_size)
                all_images.append(dcm_image)
                # patient_images.append(dcm_image)
                print(f"{folder_pair}missing nii file,number:{count}")

            elif jpg_file and not nii_file:
                
                jpg_image = load_img(jpg_file,'jpg')
                jpg_image = cv2.resize(jpg_image, target_size)
                all_images.append(jpg_image)
                # patient_images.append(jpg_image)
                print(f"{folder_pair}missing nii file,number:{count}")
                
        if not is_double:
            if len(all_images) == 2:  # 确保每个病人有 2 张图像
                # 将两个图像堆叠在一起
                patient_input = np.stack(all_images, axis=0)  # 形状为 (2, 224, 224)
                patient_images.append(patient_input)
            else:
                print(f"Skipping patient {folder_pair} due to missing images")
            
    if not is_double:
        return patient_images

    return all_images