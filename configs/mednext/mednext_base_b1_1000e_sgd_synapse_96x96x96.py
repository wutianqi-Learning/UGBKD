from mmengine.config import read_base
with read_base():
    from .mednext_small_b1_1000e_sgd_synapse_96x96x96 import * # noqa

# model settings
model.update(dict(
    backbone=dict(
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
    ),
))
