_base_ = [
    '../_base_/models/ocrnet_r50-d8.py', '../_base_/datasets/cityscapes_769x769.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained=
    'open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)),
    # test_cfg=dict(mode='whole'),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRPriorHead101',
            in_channels=2048,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=True,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0))
])

# model = dict(
#     pretrained='/home/huatang/Data/mmsegmentation/pretrained/resnet101_v1c-e67eebb6.pth',
#     backbone=dict(
#         extra=dict(
#             stage2=dict(num_channels=(48, 96)),
#             stage3=dict(num_channels=(48, 96, 192)),
#             stage4=dict(num_channels=(48, 96, 192, 384)))),
# # model training and testing settings
#     train_cfg = dict(),
#     test_cfg = dict(mode='whole'),
#     decode_head=[
#         dict(
#             type='FCNHead',
#             in_channels=[48, 96, 192, 384],
#             channels=sum([48, 96, 192, 384]),
#             input_transform='resize_concat',
#             in_index=(0, 1, 2, 3),
#             kernel_size=1,
#             num_convs=1,
#             norm_cfg=norm_cfg,
#             concat_input=False,
#             dropout_ratio=-1,
#             num_classes=150,
#             align_corners=False,
#             loss_decode=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#         dict(
#             type='OCRPriorHead',
#             in_channels=[48, 96, 192, 384],
#             channels=512,
#             ocr_channels=256,
#             input_transform='ocr_prior',
#             in_index=(0, 1, 2, 3),
#             norm_cfg=norm_cfg,
#             dropout_ratio=-1,
#             num_classes=150,
#             align_corners=False,
#             loss_decode=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_prior_decode=dict(type='AffinityLoss', loss_weight=1.0))
#     ])

