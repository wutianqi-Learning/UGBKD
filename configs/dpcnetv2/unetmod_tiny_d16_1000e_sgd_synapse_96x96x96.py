from mmengine.config import read_base
with read_base():
    from ..unet.unetmod_tiny_d16_1000e_sgd_synapse_96x96x96 import * # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2))))
