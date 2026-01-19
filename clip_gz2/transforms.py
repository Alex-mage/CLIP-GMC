import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import random


class CenterCrop:
    def __init__(self, scale_factor=0.5):
        self.scale_factor = scale_factor

    def __call__(self, img):
        width, height = img.size
        new_size = int(min(width, height) * self.scale_factor)
        new_size = max(1, new_size)
        return TF.center_crop(img, output_size=[new_size, new_size])


class RandomTranslation:
    def __init__(self, max_translation=4):
        self.max_translation = max_translation

    def __call__(self, img):
        max_dx = self.max_translation
        max_dy = self.max_translation
        dx = int((2 * torch.rand(1).item() - 1) * max_dx)
        dy = int((2 * torch.rand(1).item() - 1) * max_dy)
        return TF.affine(img, angle=0, translate=[dx, dy], scale=1.0, shear=[0])


class RandomScaling:
    def __init__(self, min_scale=1 / 1.3, max_scale=1.3):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale_factor = torch.rand(1) * (self.max_scale - self.min_scale) + self.min_scale
        return TF.resize(img, size=[int(img.height * scale_factor), int(img.width * scale_factor)])


class MorphologicalOpening:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # 将PIL图像转换为numpy数组
        img_np = np.array(img)

        # 对每个通道分别进行开操作
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if len(img_np.shape) == 3:  # RGB图像
            result = np.zeros_like(img_np)
            for i in range(img_np.shape[2]):
                result[:, :, i] = cv2.morphologyEx(img_np[:, :, i], cv2.MORPH_OPEN, kernel)
        else:  # 灰度图像
            result = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)

        # 将numpy数组转换回PIL图像
        return Image.fromarray(result)
