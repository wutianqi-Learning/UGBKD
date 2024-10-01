from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.unet.monai_unet_mod import UNetMod
from seg.models.segmentors.monai_model import MonaiSeg


# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=[96, 96, 96],
    backbone=dict(
        type=UNetMod,
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=2),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=[96, 96, 96],
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

