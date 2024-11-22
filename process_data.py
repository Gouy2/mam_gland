import os
from load_img import load_dcm, load_nii
import numpy as np


# 遍历病人的姓名，找到对应的文件夹并加载图像
def process_images_for_patients(base_path, target_size=(512, 512)):

    # 获取所有病人文件夹
    all_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

    # 按病人分组文件夹，每两个文件夹为一个病人
    patient_folders = []
    for i in range(0, len(all_folders), 2):
        if i + 1 < len(all_folders):
            patient_folders.append([all_folders[i], all_folders[i + 1]])


    # for _, row in labels_df.iterrows():
    #     label = row['N分期']  # 获取病人标签（N分期）
        
    # 加载每个文件夹中的图像
    patient_images = []

    for folder_pair in patient_folders:
        print("1")
        all_images = []
        for folder in folder_pair:
            dcm_file = None
            nii_file = None

            # print(folder)

            # 查找对应的 .dcm 和 .nii 文件
            for file in os.listdir(os.path.join(base_path, folder)):
                if file.endswith('.dcm'):
                    dcm_file = os.path.join(base_path, folder, file)
                elif file.endswith('.nii.gz'):
                    nii_file = os.path.join(base_path, folder, file)
                           
            # print(f"dcm_file: {dcm_file}, nii_file: {nii_file}")

            if dcm_file and nii_file:
                # 读取并处理 .dcm 和 .nii 图像
                dcm_image = load_dcm(dcm_file, target_size)
                nii_mask = load_nii(nii_file, target_size)

                # 将两个图像相乘
                focused_dcm_image = dcm_image * nii_mask
                all_images.append(focused_dcm_image)

            else:
                all_images.append(load_dcm(dcm_file, target_size))
                        
        if len(all_images) == 2:  # 确保每个病人有 2 张图像

            # print(all_images[0].shape)
            # 将两个图像堆叠在一起
            patient_input = np.stack(all_images, axis=0)  # 形状为 (2, 512, 512)
            # 追加至列表中
            patient_images.append(patient_input)

        else:
            print(f"Skipping patient {folder_pair} due to missing images")

    return patient_images