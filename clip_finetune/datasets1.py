import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from PIL import Image
import numpy as np
from torchvision import transforms
import os

import h5py
from sklearn.model_selection import train_test_split
from transforms import CenterCrop, RandomTranslation, RandomScaling, MorphologicalOpening
import random
from typing import List, Dict, Iterator, Optional
from collections import defaultdict


class Galaxy10Dataset(Dataset):
    def __init__(self, images, labels, preprocess, texts, augment=False):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess
        self.augment = augment

        # 选择使用固定文本描述还是随机文本描述
        self.text_descriptions = texts

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

        # 返回图像、文本描述、标签
        return processed_image, text, torch.tensor(label, dtype=torch.long)


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


class BalancedClassSampler(Sampler):
    """
    确保每个batch包含所有类别的采样器，对样本数量少的类别进行重采样
    """

    def __init__(self, labels: np.ndarray, batch_size: int = 40, samples_per_class: int = 4,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None, seed: int = 42):
        """
        初始化平衡类别采样器
        
        参数:
            labels: 数据集的标签
            batch_size: 批次大小，默认为40
            samples_per_class: 每个类别在每个batch中的样本数，默认为4
            num_replicas: 分布式训练的副本数
            rank: 分布式训练的当前进程排名
            seed: 随机种子
        """
        self.labels = labels
        self.num_classes = len(np.unique(labels))
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        # 确保batch_size能被samples_per_class * num_classes整除
        assert batch_size == samples_per_class * self.num_classes, \
            f"batch_size ({batch_size}) 必须等于 samples_per_class ({samples_per_class}) * num_classes ({self.num_classes})"

        # 分布式训练设置
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.epoch = 0
        self.seed = seed

        # 按类别索引样本
        self.class_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)

        # 计算每个类别的样本数量
        self.samples_per_class_count = {cls: len(indices) for cls, indices in self.class_indices.items()}

        # 计算样本最多的类别的样本数量
        self.max_samples_per_class = max(self.samples_per_class_count.values())

        # 计算每个进程的样本数量
        self.num_samples_per_replica = self._get_num_samples_per_replica()
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def _get_num_samples_per_replica(self) -> int:
        """计算每个进程应该处理的样本数量，基于样本最多的类别"""
        # 计算可以创建的完整batch数量，基于样本最多的类别
        # 每个类别在每个batch中需要samples_per_class个样本
        num_full_batches = self.max_samples_per_class // self.samples_per_class

        # 为了确保每个进程处理相同数量的batch，我们需要调整
        num_full_batches_per_replica = num_full_batches // self.num_replicas

        # 确保每个进程至少有一个batch
        num_full_batches_per_replica = max(1, num_full_batches_per_replica)

        # 每个进程的样本总数
        return num_full_batches_per_replica * self.batch_size

    def __iter__(self) -> Iterator[int]:
        # 设置随机种子，确保可重复性但每个epoch不同
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 为每个类别创建随机索引列表
        indices_per_class = {}
        for cls, indices in self.class_indices.items():
            # 对于每个类别，我们需要创建足够多的索引以满足所有batch的需求
            # 如果该类别的样本数量不足，我们将进行重采样
            num_required_samples = self.num_samples_per_replica // self.batch_size * self.samples_per_class

            if len(indices) >= num_required_samples:
                # 如果样本足够，直接随机排列
                indices_per_class[cls] = torch.randperm(len(indices), generator=g).tolist()
            else:
                # 如果样本不足，进行重采样
                # 首先随机排列所有样本
                perm = torch.randperm(len(indices), generator=g).tolist()
                # 然后重复采样直到达到所需数量
                repeated_perm = []
                while len(repeated_perm) < num_required_samples:
                    repeated_perm.extend(perm)
                    # 每次重复后重新打乱
                    random.shuffle(perm)

                indices_per_class[cls] = repeated_perm[:num_required_samples]

        # 计算每个进程应处理的batch数量
        num_batches_per_replica = self.num_samples_per_replica // self.batch_size

        # 计算当前进程的起始batch索引
        start_batch = self.rank * num_batches_per_replica
        end_batch = start_batch + num_batches_per_replica

        # 用于跟踪每个类别已使用的样本数量
        class_positions = {cls: 0 for cls in self.class_indices.keys()}

        # 生成batch
        result = []
        for _ in range(start_batch, end_batch):
            batch_indices = []

            # 为每个类别选择samples_per_class个样本
            for cls in range(self.num_classes):
                cls_indices = self.class_indices[cls]
                cls_perm = indices_per_class[cls]

                # 选择当前类别的samples_per_class个样本
                for _ in range(self.samples_per_class):
                    # 获取样本索引
                    idx_in_cls = cls_perm[class_positions[cls] % len(cls_perm)]
                    sample_idx = cls_indices[idx_in_cls]
                    batch_indices.append(sample_idx)

                    # 更新位置
                    class_positions[cls] += 1

            # 将当前batch的索引添加到结果中
            result.extend(batch_indices)

        return iter(result)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于分布式训练"""
        self.epoch = epoch


def load_data_from_h5(h5_file_path, texts, preprocess, batch_size=10, num_replicas=None, rank=None,
                      augment=False, shuffle=True, split='train', train_ratio=0.8, seed=42, balanced_sampling=True,
                      samples_per_class=1, train_data_ratio=1.0):
    """
    从h5文件加载数据并创建DataLoader

    参数:
        h5_file_path: h5文件路径
        ss_dir: 标准图像目录路径
        preprocess: 预处理函数
        batch_size: 批次大小，默认为40
        num_replicas: 分布式训练的副本数
        rank: 分布式训练的当前进程排名
        augment: 是否使用数据增强
        shuffle: 是否打乱数据
        split: 数据集划分，可选'train', 'val'
        train_ratio: 训练集比例
        subset_size: 用于过拟合测试的样本数量
        seed: 随机种子
        use_random_text: 是否使用随机文本描述
        balanced_sampling: 是否使用平衡采样，确保每个batch包含所有类别
        samples_per_class: 每个类别在每个batch中的样本数，默认为4
        train_data_ratio: 使用训练数据集总量的比例，默认为1.0（使用全部数据）

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

        # 特殊约定：如果 train_data_ratio <= 0，表示启用 k-shot 模式
        # 我们用 |train_data_ratio| 的整数部分作为 k 值（负数防误用）
        if train_data_ratio <= 0:
            k_shot = int(abs(train_data_ratio))  # 例如 -1 表示 1-shot，-5 表示 5-shot
            if k_shot == 0:
                k_shot = 1  # 至少 1 个样本

            # 按类别分组
            class_indices = defaultdict(list)
            for idx, label in enumerate(dataset_labels):
                class_indices[label].append(idx)

            selected_indices = []
            np.random.seed(seed)
            for cls, indices in class_indices.items():
                if len(indices) < k_shot:
                    # 样本不足时，全部使用并重复补齐（可选）
                    chosen = indices
                    while len(chosen) < k_shot:
                        chosen.extend(indices)
                    chosen = chosen[:k_shot]
                else:
                    chosen = np.random.choice(indices, size=k_shot, replace=False)
                selected_indices.extend(chosen)

            dataset_images = dataset_images[selected_indices]
            dataset_labels = dataset_labels[selected_indices]

        # 原有逻辑：train_data_ratio > 0 且 < 1.0，按比例抽样
        elif train_data_ratio < 1.0:
            num_samples = max(1, int(len(dataset_images) * train_data_ratio))
            class_indices = defaultdict(list)
            for idx, label in enumerate(dataset_labels):
                class_indices[label].append(idx)

            selected_indices = []
            for cls, indices in class_indices.items():
                cls_samples = max(1, int(len(indices) * train_data_ratio))
                np.random.seed(seed)
                chosen = np.random.choice(indices, size=cls_samples, replace=False)
                selected_indices.extend(chosen)

            dataset_images = dataset_images[selected_indices]
            dataset_labels = dataset_labels[selected_indices]

    elif split == 'val':
        dataset_images, dataset_labels = val_images, val_labels
    else:
        raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val']")

    # 创建数据集
    dataset = Galaxy10Dataset(dataset_images, dataset_labels, preprocess,
                              texts=texts, augment=augment)

    # 创建采样器
    if balanced_sampling:
        # 使用平衡类别采样器
        sampler = BalancedClassSampler(
            dataset_labels,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed
        )
    else:
        # 使用普通的分布式采样器
        sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank,
                                     shuffle=shuffle) if num_replicas else None

    # 创建数据加载器
    return DataLoader(
        dataset,
        batch_size=batch_size if not balanced_sampling else batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=True  # 确保每个batch大小相同
    )
