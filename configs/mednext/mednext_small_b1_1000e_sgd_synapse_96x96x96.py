from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.medical_seg.mednext.MedNextV1 import MedNeXt
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

dataloader_cfg.update(
    dict(num_samples=1)
)
# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=MedNeXt,
        in_channels=1,
        n_channels=32,
        n_classes=14,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2]
    ),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))
