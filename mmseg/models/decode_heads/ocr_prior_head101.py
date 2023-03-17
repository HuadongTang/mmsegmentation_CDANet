# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.norm import build_norm_layer
from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead
import numpy as np
from ..builder import HEADS, build_loss
# from mmseg.ops import resize
class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, 512, 1, bias=False),
                                   norm_layer(512))

        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        out_ = self.conv4(F.relu_(x + out))
        return out_
class AggregationModule(nn.Module):
    """Aggregation Module"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 conv_cfg=None,
                 norm_cfg=None):
        super(AggregationModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        padding = kernel_size // 2

        self.reduce_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))

        self.t1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.t2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        self.p1 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        self.p2 = ConvModule(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)
        _, self.norm = build_norm_layer(norm_cfg, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function."""
        x = self.reduce_conv(x)
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        out = self.relu(self.norm(x1 + x2))
        return out

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
# class PriorConv(nn.Module):
#     def __init__(self, height=None, width=None):
#         super(PriorConv, self).__init__()
#
#     def forward(self, x):
#         batch_size, channel, h, w = x.size()
#         # self.prior_size = int(h/4 + 1)*int(w/4 + 1) #ade20k
#         self.prior_size = h*w #int(math.ceil(h/2)) * int(math.ceil(w/2))# cityscapes
#         # reshape_value = x.view(batch_size, -1, np.prod([h, w]))
#         # context_prior_map = torch.bmm(reshape_value.permute(0, 2, 1), reshape_value)
#         # context_prior_map = context_prior_map.permute(0, 2, 1)
#
#         self.prior_conv = nn.Sequential(
#             nn.Conv2d(
#                 512,
#                 self.prior_size,
#                 1,
#                 stride=1,
#                 padding=0,
#                 groups=1), nn.BatchNorm2d(self.prior_size)).cuda()
#
#         context_prior_map = self.prior_conv(x)
#
#         return context_prior_map

@HEADS.register_module()
class OCRPriorHead101(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, scale=1, loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0), **kwargs):
        super(OCRPriorHead101, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        # self.proir_hsize = proir_hsize
        # self.proir_wsize = proir_wsize
        # self.prior_size = np.prod([proir_hsize, proir_wsize])
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
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
        self.loss_prior_decode = build_loss(loss_prior_decode)
        # self.aggregation = AggregationModule(2048, 512,
        #                                      11, self.conv_cfg,
        #                                      self.norm_cfg)
        # up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.stripPooling = StripPooling(2048, (20, 12), nn.BatchNorm2d, up_kwargs)
        # self.prior_conv = ConvModule(
        #     512,
        #     2048,
        #     1,
        #     padding=0,
        #     stride=1,
        #     groups=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=None)
        self.prior_channels = 512
        self.intra_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.inter_conv = ConvModule(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.intra_inter_conv = ConvModule(
            1024,
            512,
            1,
            padding=0,
            stride=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.context_conv = ConvModule(
        #     1024,
        #     512,
        #     1,
        #     padding=0,
        #     stride=1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     act_cfg=self.act_cfg)
        # self.prior_conv = PriorConv()
        # self.prior_conv = nn.Sequential(
        #     nn.Conv2d(
        #         512,
        #         self.prior_size,
        #         1,
        #         stride=1,
        #         padding=0,
        #         groups=1), nn.BatchNorm2d(self.prior_size))
        self.query_conv = nn.Conv2d(in_channels=self.prior_channels, out_channels=self.prior_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.prior_channels, out_channels=self.prior_channels // 8, kernel_size=1)
    def forward(self, inputs, prev_output):
        """Forward function."""
        # inputs B H w C_0
        # H = inputs.shape[2]
        # W = inputs.shape[3]
        # if H != 512 or W != 1024:
        #     inputs = F.interpolate(inputs, (512, 1024),
        #                            mode='bilinear',
        #                            align_corners=True)
        x = self._transform_inputs(inputs)
        batch_size, channels, height, width = x.size()

        # value = self.aggregation(x)
        value = self.bottleneck(x)

        proj_query = self.query_conv(value).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(value).view(batch_size, -1, width * height)
        context_prior_map = torch.bmm(proj_query, proj_key)

        # if height != self.proir_hsize or width != self.proir_wsize:
        #     value = F.interpolate(value, (self.proir_hsize, self.proir_wsize), mode='bilinear', align_corners=True)
        # context_prior_map = self.prior_conv(value)
        # context_prior_map = context_prior_map.view(batch_size,
        #                                            self.prior_size,
        #                                            -1)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map
        value = value.view(batch_size, self.prior_channels, -1)
        value = value.permute(0, 2, 1)
        intra_context = torch.bmm(context_prior_map, value)
        intra_context = intra_context.div(np.prod([height, width]))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           int(height),
                                           int(width))
        # if height != self.proir_hsize or width != self.proir_wsize:
        #     intra_context = resize(input=intra_context, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
        intra_context = self.intra_conv(intra_context)

#############################################
        inter_context = torch.bmm(inter_context_prior_map, value)
        inter_context = inter_context.div(np.prod([height, width]))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.prior_channels,
                                           int(height),
                                           int(width))
        # if height != self.proir_hsize or width != self.proir_wsize:
        #     inter_context = resize(input=inter_context, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)
        inter_context = self.inter_conv(inter_context)
        # inter_feats = self.intra_ter(torch.cat([feats, inter_context], 1))


        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        # object_context = self.object_context_block(feats, context)

        intra_feats = self.intra_inter_conv(torch.cat([feats, intra_context], dim=1))
        inter_feats = self.intra_inter_conv(torch.cat([feats, inter_context], dim=1))
        intra_object_context = self.object_context_block(intra_feats, context)
        inter_object_context = self.object_context_block(inter_feats, context)
        output = self.bottleneck_final(torch.cat([feats, intra_object_context, inter_object_context], dim=1))

        output = self.cls_seg(output)

        return output, context_prior_map
    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs, prev_output)[0]
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
        loss.update(super(OCRPriorHead101, self).losses(seg_logit, seg_label))
        prior_loss = self.loss_prior_decode(
            context_prior_map,
            self._construct_ideal_affinity_matrix(seg_label, logit_size))
        loss['loss_prior'] = prior_loss
        return loss
