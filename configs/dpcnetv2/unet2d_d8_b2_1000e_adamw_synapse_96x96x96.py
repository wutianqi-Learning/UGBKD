# import modules
from torch.nn import BatchNorm3d, ReLU
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from seg.models.backbones import ResNetV1c
from seg.models.necks.unet import UNet_Neck
from seg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss

loss_weights = [0.5714285714285714, 0.2857142857142857, 0.14285714285714285, 0.0]

norm_cfg = dict(type=BatchNorm3d, requires_grad=True)
act_cfg = dict(type=ReLU)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ResNetV1c,
        depth=18,
        in_channels=1,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type=UNet_Neck,
        base_channels=64,
        downsamples=(True, False, False),
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    decode_head=dict(
        type=FCNHead,
        in_channels=64,
        channels=64,
        in_index=3,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=14,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[0]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[0])]),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=256,
        channels=256,
        num_convs=0,
        num_classes=9,
        in_index=1,
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        upsample_label=True,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=loss_weights[2]),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=loss_weights[2])]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
