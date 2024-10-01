from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.unet.monai_unet_mod import UNetMod
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

dataloader_cfg.update(
    dict(num_samples=2)
)

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
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
        inf_size=roi,
        sw_batch_size=2,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))
