####
# This code was directly repurposed from 
# https://github.com/facebookresearch/mae
####

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from patchify import patchify
from pos_embed import get_2d_sincos_pos_embed

class PatchifyEncoder(nn.Module):
    """ 
    """
    def __init__(self, encoder: nn.Module, img_size=224):
        super().__init__()
        self.img_size = img_size

        self.encoder = encoder

    def forward(self, x, mask_ratio):
        # embed patches
        x = patchify(x, p=self.img_size)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # prepend cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)

        return x, mask, ids_restore