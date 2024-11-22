import pydicom
import nibabel as nib
import cv2
import numpy as np

# 图像预处理函数
def load_dcm(dcm_path, target_size=(512, 512)):
    # 加载 DICOM 图像
    dicom_data = pydicom.dcmread(dcm_path)
    dcm_image = dicom_data.pixel_array.astype(np.float32)

    dcm_image_normalized = (dcm_image - np.min(dcm_image)) / (np.max(dcm_image) - np.min(dcm_image))  

    # 转换尺寸（512x512）
    dcm_image_resized = cv2.resize(dcm_image_normalized, target_size)


    return dcm_image_resized

def load_jpg(jpg_path, target_size=(512, 512)):
    # 加载 DICOM 图像
    jpg_image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    jpg_image = jpg_image.astype(np.float32)

    jpg_image_normalized = (jpg_image - np.min(jpg_image)) / (np.max(jpg_image) - np.min(jpg_image))  

    # 转换尺寸（512x512）
    jpg_image_resized = cv2.resize(jpg_image_normalized, target_size)


    return jpg_image_resized

def load_nii(nii_path, target_size=(512, 512),focus_strength=5):
    # 加载 NIfTI 图像
    nii_data = nib.load(nii_path)
    nii_image = nii_data.get_fdata()

    # 去掉第三维度
    nii_image = np.squeeze(nii_image)

    # 转置 NIfTI 图像，使其与 DICOM 图像对齐
    nii_image_transposed = np.transpose(nii_image, (1, 0))  # 转置为 (2294, 1914)

    mask = np.where(nii_image_transposed > 0, 1, 0).astype(np.uint8)

    # 调整掩码大小
    mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # 使用距离变换
    dist_transform = cv2.distanceTransform(1 - mask_resized, cv2.DIST_L2, 5)
    dist_transform_normalized = cv2.normalize(dist_transform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 将标记区域设置为 1，其余区域根据距离逐渐降低
    attention_mask = np.power(1 - dist_transform_normalized, focus_strength)
    
    return attention_mask