import os
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm
from prompts import class_descriptions3
import clip
import numpy as np

class_descriptions = [
    "An astronomical image of a galaxy with a smooth, near-spherical structure and uniform stellar distribution.",
    "An image of a galaxy with a smooth, elongated, cigar-like shape and a compact stellar profile.",
    "A view of a galaxy observed edge-on, displaying a flattened disk with a prominent central bulge.",
    "An image of a spiral galaxy without a central bar, showing distinct, loosely wound spiral arms."
]


def get_text_features(args, device):
    _, preprocess = clip.load("ViT-B/16", device=device)
    if args.dataset == 'mnist':
        texts_list = class_descriptions
    elif args.dataset == 'g10d':
        texts_list = class_descriptions3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    text_tokens = []
    for t in texts_list:
        text_token = clip.tokenize(t).to(device)
        text_tokens.append(text_token)
    text_tokens = torch.cat(text_tokens, dim=0).to(device)
    return text_tokens


def train_for_bd(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, results_dir):
    model.train()
    metrics = []  # 存储每个 epoch 的评估指标
    epochs = args.epochs
    torch.autograd.set_detect_anomaly(True)
    text_tokens = get_text_features(args, device)
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0

        progress_bar = None
        if dist.get_rank() == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f"对比学习 Epoch {epoch + 1}/{epochs}")
        else:
            progress_bar = enumerate(train_loader)

        for i, (images, texts, labels) in progress_bar:
            # 确保输入数据为FP32
            images = images.float().to(device)
            labels.to(device)

            optimizer.zero_grad()

            # 获取模型输出字典
            with torch.autocast(device_type='cuda', enabled=False):
                outputs_dict = model(images, texts)
                image_features = outputs_dict["image_features"].float()
                text_features = outputs_dict["text_features"].float()
                current_logit_scale = outputs_dict["logit_scale"]

            # 计算对比损失（使用本地特征）
            loss = criterion(
                image_features,
                text_features,
                current_logit_scale
            )

            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if dist.get_rank() == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(
                    loss=running_loss / (i + 1)
                )

        if dist.get_rank() == 0:
            print(f"对比学习 Epoch [{epoch + 1}/{epochs}], "
                  f"总损失: {running_loss / len(train_loader):.4f}")

        # 每个 epoch 结束后评估
        val_metrics = evaluate(model, val_loader, criterion, device, epoch, text_tokens, args, results_dir)

        if dist.get_rank() == 0:
            # 记录评估指标
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader)
            }

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
            data_ratio_str = str(args.data_ratio).replace('.', '')
            file_name = f"{data_ratio_str}_{args.model_type}_{args.text_p}_seed{args.seed}.csv"
            metrics_file = os.path.join(results_dir, file_name)
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(metrics_file, index=False)
            print(f"对比学习指标保存到 {metrics_file}")

            # 保存模型
            if epoch == epochs - 1:
                model_path = os.path.join(results_dir, f"clip_{data_ratio_str}_seed{args.seed}.pt")
                # torch.save(model.module.state_dict(), model_path)
                print(f"对比学习模型保存到 {model_path}")


def evaluate(model, val_loader, criterion, device, epoch, text_tokens, args, results_dir):
    """评估对比学习性能并保存最后一个 epoch 的图像特征"""
    model.eval()
    all_labels = []
    all_preds = []
    all_features = []
    val_running_loss = 0.0
    val_samples = 0

    with torch.no_grad():
        progress_bar = None
        if dist.get_rank() == 0:
            progress_bar = tqdm(enumerate(val_loader), desc="评估对比学习")
        else:
            progress_bar = enumerate(val_loader)

        for i, (images, texts, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # 获取模型输出字典
            with torch.autocast(device_type='cuda', enabled=False):
                outputs_dict = model(images, texts)
                image_features = outputs_dict["image_features"].float()
                tf2 = outputs_dict["text_features"].float()
                current_logit_scale = outputs_dict["logit_scale"]

                text_features = model.module.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 计算对比损失
            loss = criterion(
                image_features,
                tf2,
                current_logit_scale
            )

            # 累计验证损失
            val_running_loss += loss.item() * images.size(0)
            val_samples += images.size(0)

            # 计算图像与文本的相似度
            sim = (image_features @ text_features.T) * current_logit_scale

            # 获取预测类别和概率
            probs = torch.softmax(sim, dim=1)
            _, preds = torch.max(sim, dim=1)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_features.append(image_features.cpu())  # 收集图像特征

    # 合并为单个 tensor
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_features = torch.cat(all_features, dim=0)  # 合并特征

    # 确保 all_labels、all_preds 和 all_features 在 CUDA 上
    all_labels = all_labels.to(device)
    all_preds = all_preds.to(device)
    all_features = all_features.to(device)

    # 收集所有进程的数据
    world_size = dist.get_world_size()
    gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
    gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
    gathered_features = [torch.zeros_like(all_features) for _ in range(world_size)]
    gathered_val_loss = [torch.tensor([val_running_loss, val_samples], dtype=torch.float32, device=device) for _ in
                         range(world_size)]

    dist.all_gather(gathered_labels, all_labels)
    dist.all_gather(gathered_preds, all_preds)
    dist.all_gather(gathered_features, all_features)
    dist.all_gather(gathered_val_loss,
                    torch.tensor([val_running_loss, val_samples], dtype=torch.float32, device=device))

    # 合并所有进程的数据
    all_labels = torch.cat(gathered_labels, dim=0)
    all_preds = torch.cat(gathered_preds, dim=0)
    all_features = torch.cat(gathered_features, dim=0)

    metrics = None
    if dist.get_rank() == 0:
        # 计算总验证损失
        total_val_loss = sum([loss_data[0] for loss_data in gathered_val_loss])
        total_val_samples = sum([loss_data[1] for loss_data in gathered_val_loss])
        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0

        # 导入评估指标计算函数
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )

        # 转换为numpy数组
        all_labels_np = all_labels.cpu().numpy()
        all_preds_np = all_preds.cpu().numpy()

        # 计算评估指标
        metrics = {
            'val_loss': avg_val_loss,
            'accuracy': accuracy_score(all_labels_np, all_preds_np),
            'macro_precision': precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
            'macro_recall': recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
            'macro_f1': f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0),
            'weighted_precision': precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0),
            'weighted_recall': recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0),
            'weighted_f1': f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
        }

        # 打印评估结果
        print(f"\n对比学习 Epoch {epoch + 1} 评估指标:")
        print(f"验证损失: {metrics['val_loss']:.4f}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"宏平均精确率: {metrics['macro_precision']:.4f}")
        print(f"宏平均召回率: {metrics['macro_recall']:.4f}")
        print(f"宏平均F1: {metrics['macro_f1']:.4f}")

        # 在最后一个 epoch 保存图像特征、预测标签和实际标签到 CSV
        if epoch == args.epochs - 1:
            # 转换为 numpy 数组
            all_features_np = all_features.cpu().numpy()
            feature_columns = [f"feature_{i}" for i in range(all_features_np.shape[1])]
            features_df = pd.DataFrame(all_features_np, columns=feature_columns)
            features_df['predicted_label'] = all_preds_np
            features_df['true_label'] = all_labels_np

            # 保存到 CSV
            data_ratio_str = str(args.data_ratio).replace('.', '')
            features_file_name = f"features_{data_ratio_str}_seed{args.seed}.csv"
            features_file = os.path.join(results_dir, features_file_name)
            features_df.to_csv(features_file, index=False)
            print(f"图像特征及标签保存到 {features_file}")

    return metrics if dist.get_rank() == 0 else None
