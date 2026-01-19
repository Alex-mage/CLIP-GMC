import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
from datasets1 import load_data_from_h5
from datasets2 import load_data
from models import load_model
from train import train_for_bd
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
from prompts import class_names, class_descriptions3


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features, logit_scale):
        # 计算所有两两对比损失
        loss_image_text = self.pairwise_contrastive_loss(image_features, text_features, logit_scale)
        # 总损失为三部分加权平均
        total_loss = loss_image_text
        return total_loss

    def pairwise_contrastive_loss(self, features_a, features_b, logit_scale):
        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.T) * logit_scale
        # 对称对比损失
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = self.cross_entropy(logits, labels)
        loss_ba = self.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2


def main(h5_file_path, results_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--data_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_type", type=str, default="c2")
    parser.add_argument("--text_p", type=str, default="des3")
    parser.add_argument("--dataset", type=str, default="mnist")

    # 分布式参数
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # filter model
    model_dir = "/home/chd/galaxy_proj/results"
    if args.model_type:
        model_dir = model_dir + f"/{args.model_type}.pt"
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if args.text_p == "des3":
        texts = class_descriptions3
    elif args.text_p == "classn":
        texts = class_names
    else:
        raise ValueError(f"Unknown text_p: {args.text_p}")

    # 初始化分布式训练
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.local_rank)

    # 创建结果目录（仅主进程）
    if dist.get_rank() == 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model, preprocess = load_model(device=device, pretrained_model_path=model_dir)

    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # 加载数据
    if args.dataset == "g10d":
        train_loader = load_data_from_h5(
            h5_file_path,
            texts,
            preprocess,
            num_replicas=dist.get_world_size() if dist.is_initialized() else None,
            rank=args.local_rank if dist.is_initialized() else None,
            augment=True,
            shuffle=True,
            split='train',
            train_ratio=args.train_ratio,
            train_data_ratio=args.data_ratio,
            seed=args.seed
        )

        val_loader = load_data_from_h5(
            h5_file_path,
            texts,
            preprocess,
            num_replicas=dist.get_world_size() if dist.is_initialized() else None,
            rank=args.local_rank if dist.is_initialized() else None,
            augment=False,
            shuffle=False,
            split='val',
            train_ratio=args.train_ratio,
            seed=args.seed
        )
    elif args.dataset == "mnist":
        train_loader = load_data(
            preprocess,
            num_replicas=dist.get_world_size() if dist.is_initialized() else None,
            rank=args.local_rank if dist.is_initialized() else None,
            augment=True,
            shuffle=True,
            split='train',
            seed=42,
            train_data_ratio=args.data_ratio
        )

        val_loader = load_data(
            preprocess,
            num_replicas=dist.get_world_size() if dist.is_initialized() else None,
            rank=args.local_rank if dist.is_initialized() else None,
            augment=False,
            shuffle=False,
            split='val',
            train_data_ratio=args.data_ratio
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    criterion = ContrastiveLoss()
    # 创建优化器
    num_epochs = args.epochs
    num_training_steps_per_epoch = len(train_loader)
    total_training_steps = num_epochs * num_training_steps_per_epoch
    initial_lr = args.lr
    warmup_steps = int(0.1 * total_training_steps)  # 10% 的训练步数用于 warmup
    min_lr_ratio = 0
    weight_decay = 0.05
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    param_optimizer = list(model.module.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "ln_final.weight", "logit_scale"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=initial_lr,
        betas=(adam_beta1, adam_beta2)
    )

    def lr_lambda(current_step: int):
        # Warmup 阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine Annealing 衰减阶段
        # 进度从 warmup_steps 到 total_training_steps
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))

        if min_lr_ratio == 0.0:
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    # 训练模型
    train_for_bd(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, results_dir)

    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    h5_file_path = "/mydata/chd/Galaxy10_DECals.h5"
    results_dir = "/home/chd/galaxy_proj/results"

    main(h5_file_path, results_dir)
