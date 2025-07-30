from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d

# 定义一个快速高斯模糊函数，使用51x51的核和标准差50,后续用于生成模糊基线图像。
blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
# Copyright (c) SenseTime. All Rights Reserved.
import cv2
import torchvision.transforms as transforms


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        # 转换参数为与输入tensor相同的设备和数据类型
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor

        # 将mean/std从[3]转为[3,1,1]（单图像）或[1,3,1,1]（批次）
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        
        if tensor.ndim > 3:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        # 通过expand广播到输入tensor的完整形状
        mean = mean.expand(tensor.shape)
        std = std.expand(tensor.shape)
        # 执行逆标准化：output=input×σ+μ
        tensor = tensor * std + mean
        # 钳制到[0,1]范围保证有效像素值
        tensor = tensor.clamp(min=0, max=1)
        return tensor


class GroupCAM(object):
    def __init__(self, model, target_layer="layer3.2", groups=32):
        super(GroupCAM, self).__init__()
        # model：要解释的目标模型,target_layer：要提取特征的中间层,groups：分组数量（默认32）
        self.model = model
        self.groups = groups
        self.gradients = dict()
        self.activations = dict()
        # 标准化与反标准化工具初始化
        self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Nutransform = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.target_layer = target_layer
        # 精确匹配目标层并注册hook
        if "backbone" in target_layer:
            for module in self.model.model.model.backbone.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)
        if 'neckCT' in target_layer:
            for module in self.model.model.model.neckCT.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)
        if "neckCS" in target_layer:
            for module in self.model.model.model.down.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

        if "neckRT" in target_layer:
            for module in self.model.model.model.down.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

        if "neckCT" in target_layer:
            for module in self.model.model.model.down.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

        if 'fdb_head' in target_layer:
            for module in self.model.model.model.fdb_head.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)
        if 'Model.relu' in target_layer:
            for module in self.model.model.named_modules():
                print(module)
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)

    # 存储反向传播时目标层的输出梯度
    def backward_hook(self, module, grad_input, grad_output):

        self.gradients['value'] = grad_output[0]

    # 存储前向传播时目标层的输出激活
    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]     # 分类置信度
        # idx = output["idx"]
        x_crop = output["x_crop"]   # 搜索区域图像块
        b, c, h, w = x_crop.size()
        self.model.model.zero_grad()
        idx = torch.argmax(cls)
        score = cls.reshape(-1)[idx]
        score.backward(retain_graph=retain_graph)

        # 获取存储的梯度/激活数据,提取特征图尺寸[batch, channels, height, width]
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        # 计算通道重要性权重：
        # 梯度全局平均（Grad-CAM方法）,加权特征激活图
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        activations = weights * activations
        # 初始化结果存储张量
        score_saliency_map = torch.zeros((1, 1, h, w))
        # GPU设备转移
        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        # 将特征图按通道维度分groups组
        masks = activations.chunk(self.groups, 1)
        with torch.no_grad():
            x_crop = x_crop / 255.0
            x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],x_crop[:, 1, :, :][:, None, :, :], x_crop[:, 0, :, :][:, None, :, :]],dim=1)
            norm_img = self.transform_norm(x_crop)
            blur_img = blur(norm_img)
            # img = self.Nutransform(blur_img)
            img = blur_img
            img = torch.cat([img[:, 2, :, :][:, None, :, :], img[:, 1, :, :][:, None, :, :], img[:, 0, :, :][:, None, :, :]], dim=1) * 255
            base_line = self.model.model.track(img)["cls"].reshape(-1)[idx]
            for saliency_map in masks:
                saliency_map = saliency_map.sum(1, keepdims=True)
                saliency_map = F.relu(saliency_map)
                threshold = np.percentile(saliency_map.cpu().numpy(), 70)
                saliency_map = torch.where(
                    saliency_map > threshold, saliency_map, torch.full_like(saliency_map, 0))
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                blur_input = norm_img * norm_saliency_map + blur_img * (1 - norm_saliency_map)
                norm_img = self.transform_norm(blur_input)
                blur_img = blur(norm_img)
                img = blur_img
                img = torch.cat([img[:, 2, :, :][:, None, :, :], img[:, 1, :, :][:, None, :, :],
                                 img[:, 0, :, :][:, None, :, :]], dim=1) * 255
                outcls = self.model.model.track(img)["cls"].reshape(-1)[idx]
                score = outcls - base_line

                # score_saliency_map += score * saliency_map
                score_saliency_map += score * norm_saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None, None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map.cpu().data, x_crop.cpu().numpy()

    def __call__(self, input, class_idx=None, retain_graph=True):
        return self.forward(input, class_idx, retain_graph)
