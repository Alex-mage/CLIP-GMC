import os
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm


def gather_features(features):
    """收集所有 GPU 上的特征，保留梯度信息"""
    world_size = dist.get_world_size()
    if world_size == 1:
        return features

    # 创建接收缓冲区
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    # 收集所有进程的特征
    dist.all_gather(gathered_features, features)
    # 合并所有特征
    gathered_tensor = torch.cat(gathered_features)

    # 将当前进程的特征替换为原始特征，以保留梯度流
    rank = dist.get_rank()
    gathered_tensor[rank * features.size(0):(rank + 1) * features.size(0)] = features

    return gathered_tensor


def evaluate(model, val_loader, criterion, device, epoch):
    """评估对比学习性能"""
    model.eval()
    val_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        progress_bar = None
        if dist.get_rank() == 0:
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="评估对比学习")
        else:
            progress_bar = enumerate(val_loader)

        for i, (images, texts) in progress_bar:
            # 确保输入数据为FP32
            images = images.float().to(device)

            # 获取模型输出字典
            with torch.autocast(device_type='cuda', enabled=False):
                outputs_dict = model(images, texts)
                image_features = outputs_dict["image_features"].float()
                text_features = outputs_dict["text_features"].float()
                current_logit_scale = outputs_dict["logit_scale"]

            # 收集所有 GPU 上的特征
            all_image_features = gather_features(image_features)
            all_text_features = gather_features(text_features)

            # 计算对比损失（使用所有 GPU 的特征）
            loss = criterion(
                all_image_features,
                all_text_features,
                current_logit_scale
            )

            val_loss += loss.item()
            batch_count += 1

            if dist.get_rank() == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(val_loss=val_loss / (i + 1))

        # 计算平均验证损失
        avg_val_loss = val_loss / batch_count if batch_count > 0 else 0

        # 计算评估指标
        metrics = {'val_loss': avg_val_loss}

        # 打印评估结果
        if dist.get_rank() == 0:
            print(f"\n对比学习 Epoch {epoch + 1} 评估指标:")
            print(f"验证损失: {avg_val_loss:.4f}")

    return metrics if dist.get_rank() == 0 else None


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, device, results_dir, epochs=5):
    model.train()
    metrics = []  # 存储每个 epoch 的评估指标
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0

        progress_bar = None
        if dist.get_rank() == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f"对比学习 Epoch {epoch + 1}/{epochs}")
        else:
            progress_bar = enumerate(train_loader)

        for i, (images, texts) in progress_bar:
            # 确保输入数据为FP32
            images = images.float().to(device)
            optimizer.zero_grad()

            # 获取模型输出字典
            with torch.autocast(device_type='cuda', enabled=False):
                outputs_dict = model(images, texts)
                image_features = outputs_dict["image_features"].float()
                text_features = outputs_dict["text_features"].float()
                current_logit_scale = outputs_dict["logit_scale"]

            # 收集所有 GPU 上的特征
            all_image_features = gather_features(image_features)
            all_text_features = gather_features(text_features)

            # 计算对比损失（使用所有 GPU 的特征）
            loss = criterion(
                all_image_features,
                all_text_features,
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

        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)

        if dist.get_rank() == 0:
            print(f"对比学习 Epoch [{epoch + 1}/{epochs}], "
                  f"总损失: {avg_train_loss:.4f}")

        # 每个 epoch 结束后评估
        val_metrics = evaluate(model, val_loader, criterion, device, epoch)

        if dist.get_rank() == 0:
            # 记录评估指标
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss
            }

            # 将验证集指标添加到记录中
            if val_metrics:
                epoch_metrics.update(val_metrics)

            metrics.append(epoch_metrics)

            # 保存评估指标到 CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_file = os.path.join(results_dir, f"contrastive_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            print(f"对比学习指标保存到 {metrics_file}")

            # 保存模型
            if epoch == epochs - 1:
                model_path = os.path.join(results_dir, f"con_model_epoch_{epoch + 1}.pt")
                torch.save(model.module.state_dict(), model_path)
                print(f"对比学习模型保存到 {model_path}")
