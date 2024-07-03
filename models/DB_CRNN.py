# @Time : 2024-06-29 0:21
# @Author : wyj
# @File : DB_CRNN.py
# @Describe :
import math

import cv2
import hydra
import torch.nn as nn
import torch
import numpy as np
from models.bridge import Bridge
from src.utils import crop_and_resize
from .adapter import Adapter
from .DBnet.DBnet import DBTextModel
from .Rcnn.crnn import CRNN
import os
from src.losses import DBLoss
from functools import partial
class DB_CRNN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # TODO 加载DBnet模型
        self.device = torch.device(cfg.meta.device)
        self.DBTextModel = DBTextModel().to(self.device)
        # TODO 对主干网络的参数进行冻结,DB的主干网络包括resnet13和FPN结构,最后的输出为(1,256,56,56)
        for name, d in self.DBTextModel.named_parameters():
            # print(name)
            d.requires_grad = False
            if "backbone" in name or "segmentation_body" in name:
                if "bn" in name:
                    d.requires_grad = True
                else:
                    d.requires_grad = False
        # TODO 加载DBnet预训练模型
        assert os.path.exists(cfg.model.DBnet)
        self.DBTextModel.load_state_dict(torch.load(cfg.model.DBnet,
                                                    map_location=self.device))
        self.DBTextModel.adapter = nn.ModuleList([Adapter(256) for i in range(1)])
        self.DBTextModel.adapter.to(self.device)
        # TODO 加载CRNN预训练模型
        self.CRNN = CRNN(cfg).to(self.device)
        # TODO 加载预训练权重
        assert os.path.exists(cfg.model.CRNN)
        state_dict = torch.load(cfg.model.CRNN, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        self.CRNN.load_state_dict(new_state_dict)
        # TODO 冻结模型参数
        for name, d in self.CRNN.named_parameters():
            d.requires_grad = False
        self.bridge = Bridge(img_size=(32, 128), patch_size=8, embed_dim=512, depth=1, num_heads=8, mlp_ratio=4,
                             qkv_bias=True,
                             in_chans=256,norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.bridge.to(self.device)
        rec_mean = rec_std = torch.tensor(0.5)
        self.normalizer2 = lambda x: (x / 255 - rec_mean) / rec_std
        self.rec_criterion = nn.CTCLoss(blank=0, reduction='mean')
        self.dec_criterion = DBLoss(alpha=cfg.optimizer.alpha,
                       beta=cfg.optimizer.beta,
                       negative_ratio=cfg.optimizer.negative_ratio,
                       reduction=cfg.optimizer.reduction).to(self.device)
        self.CRNN.adapter = nn.ModuleList([Adapter(63) for i in range(1)])
        self.CRNN.adapter.to(self.device)
        self.nnNorm = nn.LayerNorm(512)
        self.nnNorm.to(self.device)
    def forward(self, batch,istrian=True):
        # TODO 进来的是一个batch
        feature,preds = self.DBTextModel(batch['img'],istrian)
        _batch = torch.stack([
            batch['prob_map'], batch['supervision_mask'],
            batch['thresh_map'], batch['text_area_map']
        ])
        loss = {}
        loss['preds'] = preds
        if istrian:
            prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss = self.dec_criterion(
                preds, _batch)
            loss['prob_loss'] = prob_loss
            loss['threshold_loss'] = threshold_loss
            loss['binary_loss'] = binary_loss
            loss['prob_threshold_loss'] = prob_threshold_loss
            loss['total_loss'] = total_loss
        else:
            total_loss = self.dec_criterion(
                preds, _batch)
            loss['total_loss'] = total_loss
        # TODO 声明一个空的二维列表
        all_crop = []
        feature_crop = []
        gt = batch["gt"]
        for i in range(len(batch["image_origin"])):
            # 声明一个当前图像的子列表
            length = batch["lengths"][i].shape[0]
            for j in range(length):
                points = np.array(gt[i][j],dtype=np.int32)
                data = crop_and_resize(self.normalizer2(batch["image_origin"][i]),points,type = 1)
                data2 = crop_and_resize(feature[i].detach(),points)
                all_crop.append(data)
                feature_crop.append(data2)
            # 将当前图像的子列表添加到主列表中
        all_crop = torch.stack(all_crop)
        feature_crop = torch.stack(feature_crop)
        texts = torch.stack([dd for data in batch["texts"] for dd in data])
        lengths = torch.stack([dd for data in batch["lengths"] for dd in data])
        data = (all_crop,texts,lengths,feature_crop)
        # TODO 将数据输入到CRNN中
        preds = self.CRNN(data,self.bridge,self.nnNorm)
        T, N, C = preds.shape
        input = preds.log_softmax(2)
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        cost = self.rec_criterion(input, texts, input_lengths, lengths)
        loss['total_loss'] +=cost
        print("--------------------------------------------")
        try:
            # 检查 total_loss 是否为无穷大
            if math.isinf(total_loss):
                raise ValueError("total_loss is infinite")
            # 检查 cost 是否为无穷大
            if math.isinf(cost):
                raise ValueError("cost is infinite")
            # 如果没有异常，打印 total_loss 和 cost
            print(total_loss, cost)

        except ValueError as e:
            # 处理 ValueError 异常
            print("An error occurred:", e)
        print("---------------------------------------------")
        loss['preds_dec'] = preds
        return loss
# @hydra.main(config_path="../myconfig.yaml")
# def run(cfg):
#     model = DB_CRNN(cfg)
#     data = np.load('/root/autodl-fs/DB_CRNN/models/aaaa.npy',allow_pickle=True).item()
#     model(data)
# if __name__ == '__main__':
#     run()