import torch
from torch.utils.data import Dataset



class ImageDataset(Dataset):
    def __init__(self, patient_images, transform=None):
        """
        初始化数据集
        :param patient_images: 包含 (patient_input, label) 的列表
        :param transform: 数据增强
        """
        self.patient_images = patient_images
        self.transform = transform

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.patient_images)

    def __getitem__(self, idx):
        """
        返回指定索引的样本和标签
        :param idx: 索引
        :return: patient_input, label
        """
        patient_input, label = self.patient_images[idx]

        # 将 patient_input 转换为张量
        patient_input = torch.tensor(patient_input, dtype=torch.float32)

        # 应用数据增强
        if self.transform:
            patient_input = self.transform(patient_input)

        # 返回样本和标签
        return patient_input, torch.tensor(label, dtype=torch.long)
    

class ImageDataset2(Dataset):
    def __init__(self, patient_images, transform=None):
        """
        初始化数据集
        :param patient_images: 包含 (patient_input, label) 的列表
        :param transform: 数据增强
        """
        self.patient_images = patient_images
        self.transform = transform

    def __len__(self):
        """返回数据集的样本总数"""
        return len(self.patient_images)

    def __getitem__(self, idx):
        """
        返回指定索引的样本和标签
        :param idx: 索引
        :return: patient_input, label
        """
        patient_input, label = self.patient_images[idx]

        # 将 patient_input 转换为张量
        patient_input = torch.tensor(patient_input, dtype=torch.float32)

        patient_input = torch.unsqueeze(patient_input, 0)  # 变为(1,224,224)

        #变为(3,224,224)
        patient_input = torch.cat([patient_input, patient_input, patient_input], 0)

        # 应用数据增强
        if self.transform:
            patient_input = self.transform(patient_input)

        # 返回样本和标签
        return patient_input, torch.tensor(label, dtype=torch.long)