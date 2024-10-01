from mmengine.config import read_base
with read_base():
    from .unetmod_tiny_d8_b1_1000e_sgd_synapse_96x96x96 import * # noqa

# model settings
model.update(
    dict(
        backbone=dict(
            num_res_units=1)))