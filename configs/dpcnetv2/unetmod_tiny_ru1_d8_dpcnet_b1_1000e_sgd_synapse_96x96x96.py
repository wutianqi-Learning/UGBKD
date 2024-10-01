from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V14
with read_base():
    from .unetmod_tiny_ru1_d8_b1_1000e_sgd_synapse_96x96x96 import * # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            plugin=dict(type=DSA_V14))))