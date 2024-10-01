#  https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/Pretrain/models/ssl_head.py
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SSLHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feature_size=48,
                 upsample="vae", 
                 dim=768,
                 dropout_path_rate=0.,
                 use_checkpoint=False,
                 spatial_dims=2):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose2d(dim, in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=(2, 2), stride=(2, 2)),
                nn.ConvTranspose2d(dim // 16, in_channels, kernel_size=(2, 2), stride=(2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim // 16, in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        _, c, h, w = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=3)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=3)
        x_rec = x_rec.view(-1, c, h, w)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec