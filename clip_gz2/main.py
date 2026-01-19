import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
# 导入自定义的函数和模块
from datasets1 import load_data
from models import load_model
from train import train
from lossfuc import ContrastiveLoss


def main(data_dir, results_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)

    # 分布式参数
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # 初始化分布式训练
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.local_rank)

    # 创建结果目录（仅主进程）
    if dist.get_rank() == 0 and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model, preprocess = load_model(device=device, trainable_temperature=True)

    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # 加载数据
    train_loader = load_data(
        data_dir=data_dir,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_replicas=dist.get_world_size() if dist.is_initialized() else None,
        rank=args.local_rank if dist.is_initialized() else None,
        shuffle=True,
    )

    val_loader = load_data(
        data_dir=data_dir,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_replicas=dist.get_world_size() if dist.is_initialized() else None,
        rank=args.local_rank if dist.is_initialized() else None,
        augment=False,
        shuffle=False,
    )

    criterion = ContrastiveLoss()
    # 创建优化器
    num_epochs = args.epochs
    num_training_steps_per_epoch = len(train_loader)
    total_training_steps = num_epochs * num_training_steps_per_epoch
    initial_lr = args.lr
    warmup_steps = int(0.1 * total_training_steps)  # 10% 的训练步数用于 warmup
    min_lr_ratio = 0.01
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
    train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, results_dir,
          epochs=args.epochs)

    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    data_dir = "/mydata/chd/balanced_data01"
    results_dir = "/home/chd/galaxy_proj/results"
    main(data_dir, results_dir)
