# copy right from:
# https://github.com/sacmehta/EdgeNets/blob/master/model/segmentation/espnetv2.py
from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.nets.espnetv2 import ESPNetv2Segmentation
from seg.models.segmentors.monai_model import MonaiSeg
from seg.models.losses.edl_loss2 import EvidenceLoss2
with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=ESPNetv2Segmentation,
        classes=14,
        args=dict(
            s=1.0,
            channels=1,
            num_classes=14)),
    loss_functions=dict(
        type=EvidenceLoss2, epoch=1),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='enet-sgd-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
