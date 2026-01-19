import torch
import torch.nn as nn
import clip


class ModifiedCLIPModel(nn.Module):

    def __init__(self, model_name="ViT-B/16", device="cuda", trainable_temperature=True, fixed_temperature=1.0, freeze_text_encoder=False):
        super().__init__()
        # 加载原始CLIP模型
        clip_model, _ = clip.load(model_name, device=device)

        # 保存所有组件（保持可训练状态）
        self.visual = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # 设置temperature参数
        if trainable_temperature:
            self.logit_scale = clip_model.logit_scale  # 使用CLIP原始的可训练logit_scale
        else:
            # 创建一个固定值的参数（不需要梯度）
            self.logit_scale = nn.Parameter(torch.tensor(fixed_temperature).log(), requires_grad=False)

        # 文本编码参数
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size

        # 如果需要冻结文本编码器，将相关参数的requires_grad设为False
        if freeze_text_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.token_embedding.parameters():
                param.requires_grad = False
            # 直接设置positional_embedding的requires_grad属性
            self.positional_embedding.requires_grad = False
            for param in self.ln_final.parameters():
                param.requires_grad = False
            self.text_projection.requires_grad = False

        self.to(device)

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x += self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Explicitly cast to the model's dtype before the transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # Ensure the projection layer input matches its weight dtype
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images, text):
        # 图像处理
        images = images
        image_features = self.encode_image(images)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalized features

        # 处理文本输入
        text_tokens = clip.tokenize(text).to(images.device)
        text_features = self.encode_text(text_tokens)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalized features

        logit_scale = self.logit_scale.exp().item()

        return {
            "image_features": image_features_norm,
            "text_features": text_features_norm,
            "logit_scale": logit_scale
        }


def load_model(device="cuda", trainable_temperature=True, fixed_temperature=100.0, freeze_text_encoder=False):
    _, preprocess = clip.load("ViT-B/16", device=device)

    # 实例化可微调模型
    model = ModifiedCLIPModel(
        model_name="ViT-B/16",
        device=device,
        trainable_temperature=trainable_temperature,
        fixed_temperature=fixed_temperature,
        freeze_text_encoder=freeze_text_encoder
    )
    model = model.float()

    return model, preprocess
