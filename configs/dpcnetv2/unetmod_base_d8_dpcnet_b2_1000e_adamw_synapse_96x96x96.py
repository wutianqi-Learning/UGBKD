from mmengine.config import read_base
from seg.models.utils.dsa import DSA_V14

with read_base():
    from .unetmod_base_d8_b2_1000e_adamw_synapse_96x96x96 import *  # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            plugin=dict(type=DSA_V14))))