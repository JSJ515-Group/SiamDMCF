import torch
from torch import nn
import math
from pysot.core.config import cfg
import torch.nn.functional as F
from pysot.attention.SCCA import SCCA

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class DMCFHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, add_mean=True):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DMCFHead, self).__init__()
        self.task_aware_cls = SCCA(dim=in_channels, head_num=16)
        self.task_aware_reg = SCCA(dim=in_channels, head_num=16)

        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES
        # 每个目标的最大回归数
        self.reg_max = cfg.TRAIN.REG_MAX
        # 最多的候选回归框数量
        self.reg_topk = cfg.TRAIN.REG_TOPK
        # 设置回归输出的维度
        self.total_dim = cfg.TRAIN.REG_TOPK
        self.add_mean = add_mean
        # 如果 add_mean 为 True，增加一个均值维度
        if add_mean:
            self.total_dim += 1
        # 存储分类和边界框回归部分的卷积层
        cls_tower = []
        bbox_tower = []
        self.Scale = Scale(1.0)
        # 在一个循环中，基于 NUM_CONVS（卷积层的数量）添加卷积层
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        # 回归置信度的网络部分，通常用于计算每个位置是否包含目标
        self.reg_conf = nn.Sequential(
            nn.Conv2d(4*self.total_dim,64,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1),
            nn.Sigmoid()
        )
        # 通过 add_module 方法将分类部分（cls_tower）和边界框回归部分（bbox_tower）添加到模型中
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        # 用于计算类别的卷积层，输出的通道数为 num_classes，即类别数量
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3,stride=1,padding=1)
        # 用于计算边界框预测的卷积层
        self.bbox_pred = nn.Conv2d(in_channels, 4 * (self.reg_max + 1), kernel_size=3, stride=1, padding=1)
        self.cls_sigmoid = nn.Sigmoid()
        # 对 reg_conf 中的卷积层权重进行正态分布初始化，标准差为 0.01，对偏置项初始化为 0。
        for l in self.reg_conf:
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)
        # initialization
        # 对 cls_tower、bbox_tower、cls_logits 和 bbox_pred 中的所有卷积层进行初始化，
        # 权重用标准差为 0.01 的正态分布初始化，偏置项初始化为 0。
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # initialize the bias for focal loss
        # prior_prob 是先验概率,初始化了 cls_logits 层的偏置
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        # 对 bbox_pred 的卷积层权重进行正态分布初始化，标准差为 0.01，偏置项初始化为 0。
        torch.nn.init.normal_(self.bbox_pred.weight, std=0.01)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)


    def forward(self, xc, xr):
        xc = self.task_aware_cls(xc)
        xr = self.task_aware_reg(xr)
        cls_tower = self.cls_tower(xc)
        logits = self.cls_sigmoid(self.cls_logits(cls_tower))
        bbox_reg = self.bbox_pred(self.bbox_tower(xr))
        # 如果 REG_MAX 为 0，说明模型采用直接回归的方式（而非离散分类的方式）
        if cfg.TRAIN.REG_MAX == 0:
            # 对预测值取指数，确保回归值为正数
            bbox_reg = torch.exp(bbox_reg)
        # 若启用 IoU 指导机制
        elif cfg.TRAIN.IOUGUIDED:
            # 将每个回归量分解为多个离散类别的概率分布,然后通过 Softmax 得到概率分布prob
            N,C,H,W = bbox_reg.size()
            prob = F.softmax(bbox_reg.reshape(N,4,self.reg_max+1,H,W),dim=2)
            # 取出概率分布中最大的 topk 个值，用于后续计算
            prob_topk,_ = prob.topk(self.reg_topk, dim=2)
            # 通过拼接 topk 概率和其均值，统计信息用于回归分数的进一步处理
            if self.add_mean:
                stat = torch.cat([prob_topk,prob_topk.mean(dim=2, keepdim=True)],dim=2).to(bbox_reg.device)
            else:
                stat = prob_topk
            # 通过 self.reg_conf 模块对 stat 进行卷积，生成用于回归调整的 IoU 引导分数qs
            qs = self.reg_conf(stat.reshape(N,-1,H,W))
            # 如果是二分类问题（背景和目标），分类结果经过调整
            # background：背景类别概率。foreground：前景类别概率，结合 IoU 引导分数qs进行加权。
            # 将调整后的背景和前景概率拼接，形成新的分类结果。
            if cfg.TRAIN.NUM_CLASSES == 2:
                background = logits[:,0,:,:].reshape(N,-1,H,W)
                foreground = logits[:,-1,:,:].reshape(N,-1,H,W) * qs
                logits = torch.cat([background,foreground],dim=1)
            else:
                # 如果是多分类问题，直接将分类 logits 和 IoU 引导分数qs 相乘
                logits = logits * qs

        return logits, bbox_reg




