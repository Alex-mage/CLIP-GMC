import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from transforms import CenterCrop, RandomTranslation, RandomScaling, MorphologicalOpening


class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, preprocess, augment=True, indices=None):
        self.data = pd.read_csv(csv_file, dtype={0: int})
        # 如果提供了索引，则只使用这些索引的数据
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.augment = augment
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
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx, 0]
        image_path = os.path.join(self.data_dir, f"{image_name}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.augment and self.augment_transforms is not None:
            image = self.augment_transforms(image)
        image = self.preprocess(image)
        text = self.data.iloc[idx, 1]
        return image, text


def load_data(data_dir, preprocess, batch_size=32, num_replicas=None, rank=None, augment=True, shuffle=True, split='train', val_ratio=0.01, random_state=42):
    csv_file = os.path.join(data_dir, "match.csv")
    
    # 读取CSV文件以获取数据总量
    data = pd.read_csv(csv_file, dtype={0: int})
    indices = np.arange(len(data))
    
    # 使用train_test_split进行数据集划分
    train_indices, val_indices = train_test_split(
        indices, test_size=val_ratio, random_state=random_state, shuffle=True
    )
    
    # 根据split参数选择相应的索引
    if split == 'train':
        selected_indices = train_indices
        use_augment = augment  # 训练集使用数据增强
    elif split == 'val':
        selected_indices = val_indices
        use_augment = False   # 验证集不使用数据增强
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
    
    # 创建数据集，传入选定的索引
    dataset = CustomDataset(csv_file, data_dir, preprocess, augment=use_augment, indices=selected_indices)
    
    # 创建数据加载器
    sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank,
                               shuffle=shuffle) if num_replicas else None
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                     shuffle=(shuffle and sampler is None))
