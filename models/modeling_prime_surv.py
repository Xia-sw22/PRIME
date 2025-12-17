from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

from models.embed import Embeddings
from models.encoder_parallel import EncoderParallel
from models.configs import get_PRIME_config


logger = logging.getLogger(__name__)


class TransformerBackbone(nn.Module):
    """PRIME-PVTT 的 Transformer 主干，包含 ETS / ES 双通道结构。

    同时返回编码后的序列及各模态 token 长度，方便上层进行多模态池化与对比学习。
    """

    def __init__(self, config, img_size, vis):
        super(TransformerBackbone, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = EncoderParallel(config, vis)

    def forward(self, input_ids, cc=None, lab=None, sex=None, age=None, cohort=None):
        # 多模态嵌入
        embedding_output, cc, lab, sex, age, cohort = self.embeddings(input_ids, cc, lab, sex, age, cohort)
        text = cc
        clinical = torch.cat((lab, sex, age), 1)

        # 记录各模态长度，后续用于从 encoder 输出中切分
        img_len = embedding_output.size(1)
        text_len = text.size(1)
        clinical_len = clinical.size(1)

        encoded, attn_weights = self.encoder(embedding_output, text, clinical, cohort)
        return encoded, attn_weights, img_len, text_len, clinical_len


class PRIME(nn.Module):
    """
    PRIME-PVTT 主模型。

    - 支持生存任务 (num_classes=1)
    - 也可用于分类任务 (num_classes>1)
    """

    def __init__(self, config=None, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(PRIME, self).__init__()
        if config is None:
            config = get_PRIME_config()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = TransformerBackbone(config, img_size, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)

        # 对比学习投影头（与 IRENE 类似），用于输出 image/text/clinical 的低维对比特征
        self.projection_dim = 128
        self.image_proj = nn.Sequential(
            nn.Linear(config.hidden_size, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.hidden_size, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        self.clinical_proj = nn.Sequential(
            nn.Linear(config.hidden_size, self.projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

    def forward(self, x, cc=None, lab=None, sex=None, age=None, cohort=None, labels=None):
        # 经 ETS/ES 双通道编码
        encoded, attn_weights, img_len, text_len, clinical_len = self.transformer(x, cc, lab, sex, age, cohort)

        # 按模态切分 token 序列
        image_tokens = encoded[:, :img_len, :]
        text_tokens = encoded[:, img_len : img_len + text_len, :]
        clinical_tokens = encoded[:, img_len + text_len :, :]

        # 平均池化得到模态级特征
        image_features = image_tokens.mean(dim=1)
        text_features = text_tokens.mean(dim=1)
        clinical_features = clinical_tokens.mean(dim=1)

        # 投影到对比学习空间
        image_embeds = self.image_proj(image_features)
        text_embeds = self.text_proj(text_features)
        clinical_embeds = self.clinical_proj(clinical_features)

        # 分类 / 生存任务共享的 head：对所有 token 做平均池化
        pooled = encoded.mean(dim=1)
        logits = self.head(pooled)

        # 与 IRENE 保持类似接口：返回 logits, attn, 以及三种模态的对比 embedding
        return logits, attn_weights, image_embeds, text_embeds, clinical_embeds


