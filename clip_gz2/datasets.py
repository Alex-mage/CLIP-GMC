import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import numpy as np
from torchvision import transforms
import os

import h5py
from sklearn.model_selection import train_test_split
from transforms import CenterCrop, RandomTranslation, RandomScaling, MorphologicalOpening
# 导入prompts模块以获取文本描述
from prompts import class_descriptions, class_descriptions_random


class Galaxy10Dataset(Dataset):
    def __init__(self, images, labels, preprocess, ss_dir=None, augment=False, use_random_text=False):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess
        self.augment = augment
        self.ss_dir = ss_dir
        # 选择使用固定文本描述还是随机文本描述
        self.text_descriptions = class_descriptions_random if use_random_text else class_descriptions
        
        if self.augment:
            self.augment_transforms = transforms.Compose([
                MorphologicalOpening(kernel_size=3),
                CenterCrop(scale_factor=0.8),
                transforms.RandomRotation(degrees=360),
                RandomTranslation(max_translation=4),
                RandomScaling(min_scale=1 / 1.3, max_scale=1.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(contrast=0.5)
            ])
        else:
            self.augment_transforms = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 将NumPy数组转换为PIL图像
        image = Image.fromarray(image)

        if self.augment and self.augment_transforms is not None:
            image = self.augment_transforms(image)

        # 预处理图像
        processed_image = self.preprocess(image)
        
        # 获取对应标签的文本描述
        text = self.text_descriptions[label]
        
        # 获取标准图像
        if self.ss_dir:
            standard_image_path = os.path.join(self.ss_dir, f"{label}.png")
            if os.path.exists(standard_image_path):
                standard_image = Image.open(standard_image_path).convert('RGB')
                standard_image = self.preprocess(standard_image)
            else:
                # 如果标准图像不存在，使用零张量代替
                standard_image = torch.zeros_like(processed_image)
        else:
            standard_image = torch.zeros_like(processed_image)

        # 返回图像、标签、文本描述和标准图像
        return processed_image, standard_image, text, torch.tensor(label, dtype=torch.long)


def load_galaxy10_data(h5_file_path):
    """
    从h5文件加载Galaxy10 DECaLS数据集

    参数:
        h5_file_path: h5文件路径

    返回:
        images: 图像数据
        labels: 标签数据
    """
    with h5py.File(h5_file_path, 'r') as f:
        images = np.array(f['images'])
        labels = np.array(f['ans'])

    return images, labels


def split_galaxy10_data(images, labels, train_ratio=0.8, seed=42):
    """
    划分Galaxy10 DECaLS数据集为训练集和验证集

    参数:
        images: 图像数据
        labels: 标签数据
        train_ratio: 训练集比例
        seed: 随机种子

    返回:
        train_images, train_labels: 训练集
        val_images, val_labels: 验证集
    """
    # 直接划分为训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, train_size=train_ratio, random_state=seed, stratify=labels
    )

    return train_images, train_labels, val_images, val_labels


# 在load_data_from_h5函数中添加subset_size参数
def load_data_from_h5(h5_file_path, ss_dir, preprocess, batch_size=32, num_replicas=None, rank=None, 
                     augment=False, shuffle=True, split='train', train_ratio=0.8, subset_size=None,
                     seed=42, use_random_text=False):
    """
    从h5文件加载数据并创建DataLoader

    参数:
        h5_file_path: h5文件路径
        ss_dir: 标准图像目录路径
        preprocess: 预处理函数
        batch_size: 批次大小
        num_replicas: 分布式训练的副本数
        rank: 分布式训练的当前进程排名
        augment: 是否使用数据增强
        shuffle: 是否打乱数据
        split: 数据集划分，可选'train', 'val'
        train_ratio: 训练集比例
        subset_size: 用于过拟合测试的样本数量
        seed: 随机种子
        use_random_text: 是否使用随机文本描述

    返回:
        dataloader: 数据加载器
    """
    # 加载数据
    images, labels = load_galaxy10_data(h5_file_path)

    # 划分数据集
    train_images, train_labels, val_images, val_labels = split_galaxy10_data(
        images, labels, train_ratio, seed
    )

    # 根据split参数选择相应的数据集
    if split == 'train':
        dataset_images, dataset_labels = train_images, train_labels
    elif split == 'val':
        dataset_images, dataset_labels = val_images, val_labels
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val']")
    
    # 如果指定了subset_size，则只取前subset_size个样本
    if subset_size is not None and subset_size > 0:
        dataset_images = dataset_images[:min(subset_size, len(dataset_images))]
        dataset_labels = dataset_labels[:min(subset_size, len(dataset_labels))]

    # 创建数据集
    dataset = Galaxy10Dataset(dataset_images, dataset_labels, preprocess, 
                             ss_dir=ss_dir, augment=augment, use_random_text=use_random_text)

    # 创建采样器
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank,
                                 shuffle=shuffle) if num_replicas else None

    # 创建数据加载器
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(shuffle and sampler is None))
