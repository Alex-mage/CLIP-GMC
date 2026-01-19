import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from datasets1 import load_data_from_h5
from datasets2 import load_data
from models import load_model
import pandas as pd
from prompts import class_names, class_descriptions3
from train import evaluate, get_text_features
import torch.nn as nn


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

    parser.add_argument("--model_type", type=str, default="c3")
    parser.add_argument("--text_p", type=str, default="des3")
    parser.add_argument("--dataset", type=str, default="g10d")

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

    # 训练模型
    text_tokens = get_text_features(args, device)
    metrics = []  # 存储每个 epoch 的评估指标
    # 每个 epoch 结束后评估
    val_metrics = evaluate(model, val_loader, criterion, device, 0, text_tokens, args, results_dir)

    if dist.get_rank() == 0:
        # 记录评估指标
        epoch_metrics = {}
        # 添加验证损失（如果存在）
        if val_metrics and 'val_loss' in val_metrics:
            epoch_metrics['val_loss'] = val_metrics['val_loss']

        # 添加分类评估指标（如果存在）
        for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                       'weighted_precision', 'weighted_recall', 'weighted_f1']:
            if metric in val_metrics:
                epoch_metrics[metric] = val_metrics[metric]

        metrics.append(epoch_metrics)

        # 保存评估指标到 CSV
        metrics_df = pd.DataFrame(metrics)
        data_ratio_str = str(args.data_ratio).replace('.', '')
        file_name = f"zero_{args.dataset}_seed{args.seed}.csv"
        metrics_file = os.path.join(results_dir, file_name)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"对比学习指标保存到 {metrics_file}")

    # 清理分布式环境
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    h5_file_path = "/mydata/chd/Galaxy10_DECals.h5"
    results_dir = "/home/chd/galaxy_proj/results"

    main(h5_file_path, results_dir)
