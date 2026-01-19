import torch
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
