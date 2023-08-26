# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import numpy as np
import torch.nn.functional as F
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from ..builder import HEADS, build_loss

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context
class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self,  query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        # output = context
        # output = self.bottleneck(torch.cat([context, feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return context
@HEADS.register_module()
class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), scale=1, loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.scale = scale
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.intra_conv = ConvModule(
            512,
            512,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inter_conv = ConvModule(
            512,
            512,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.intra_feats_conv = ConvModule(
            1024,
            512,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inter_feats_conv = ConvModule(
            1024,
            512,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)
        self.object_context_block = ObjectAttentionBlock(
            512,
            256,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck_final = ConvModule(
            1536,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.query_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.loss_prior_decode = build_loss(loss_prior_decode)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels):
            if i != 1:
                fpn_outs[i] = resize(
                    fpn_outs[i],
                    size=fpn_outs[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)

        batch_size, channels, height, width = output.size()
        value = output
        proj_query = self.query_conv(value).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(value).view(batch_size, -1, width * height)
        context_prior_map = torch.bmm(proj_query, proj_key)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map
        value = value.view(batch_size, 512, -1)
        value = value.permute(0, 2, 1)
        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod([height, width]))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, 512,
                                           int(height),
                                           int(width))
        intra_context = self.intra_conv(intra_context)
        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod([height, width]))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, 512,
                                           int(height),
                                           int(width))
        inter_context = self.inter_conv(inter_context)

        prev_output = self.cls_seg(output)
        feats = output
        context = self.spatial_gather_module(feats, prev_output)
        intra_feats = self.intra_feats_conv(torch.cat([feats, intra_context], dim=1))
        inter_feats = self.inter_feats_conv(torch.cat([feats, inter_context], dim=1))
        intra_object_context = self.object_context_block(intra_feats, context)
        inter_object_context = self.object_context_block(inter_feats, context)
        output = self.bottleneck_final(torch.cat([feats, intra_object_context, inter_object_context], dim=1))
        output = self.cls_seg(output)

        return output, context_prior_map

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)[0]

    def _construct_ideal_affinity_matrix(self, label, label_size):
        scaled_labels = F.interpolate(
            label.float(), size=label_size, mode="nearest")
        scaled_labels = scaled_labels.squeeze_().long()
        scaled_labels[scaled_labels == 255] = self.num_classes
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))
        return ideal_affinity_matrix
    def losses(self, seg_logit, seg_label):
        """Compute ``seg``, ``prior_map`` loss."""
        seg_logit, context_prior_map = seg_logit
        # logit_size = seg_logit.shape[2:]
        logit_size = (int(seg_logit.size(2)), int(seg_logit.size(3)))
        # logit_size = (int(seg_logit.size(2)/4), int(seg_logit.size(3)/4))
        loss = dict()
        loss.update(super(UPerHead, self).losses(seg_logit, seg_label))
        prior_loss = self.loss_prior_decode(
            context_prior_map,
            self._construct_ideal_affinity_matrix(seg_label, logit_size))
        loss['loss_prior'] = prior_loss
        return loss
