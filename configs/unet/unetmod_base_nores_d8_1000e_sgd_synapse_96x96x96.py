from mmengine.config import read_base
from seg.models.unet.monai_unet_mod import UNetMod
with read_base():
    from .unet_base_sgd_synapse import *   # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            type=UNetMod,
            channels=(64, 128, 256, 512),
            strides=(2, 2, 2),
            num_res_units=0)))
