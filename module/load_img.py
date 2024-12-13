import pydicom
import nibabel as nib
import cv2
import numpy as np
from skimage import exposure

# 图像预处理函数
def load_dcm(dcm_path):
    # 加载 DICOM 图像
    dicom_data = pydicom.dcmread(dcm_path)
    dcm_image = dicom_data.pixel_array.astype(np.float32)

    dcm_image = (dcm_image - np.min(dcm_image)) / (np.max(dcm_image) - np.min(dcm_image))  

    dcm_image = exposure.equalize_adapthist(dcm_image, clip_limit=0.05)

    # 转换尺寸（512x512）
    # dcm_image_resized = cv2.resize(dcm_image, target_size)


    return dcm_image

def load_jpg(jpg_path, target_size=(512, 512)):
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