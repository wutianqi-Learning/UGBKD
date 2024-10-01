from mmengine.config import read_base
from monai.losses import DiceCELoss
from seg.models.transunet3d.transunet3d_model import Generic_TransUNet_max_ppbp as TransUNet
from seg.models.segmentors.monai_model import MonaiSeg

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=TransUNet,
        patch_size=roi,
        base_num_features=32,
        num_pool=5,
        input_channels=1,
        num_classes=14,
        is_masked_attn=True,
        max_dec_layers=3,
        is_max_bottleneck_transformer=True,
        vit_depth=12,
        max_msda='',
        is_max_ms=True, # num_feature_levels= 3; default fpn downsampled to os244
        max_ms_idxs=[-4, -3, -2],
        max_hidden_dim=192,
        mw=1.0,
        is_max_ds=True,
        is_masking=True,
        is_max_hungarian=True,
        num_queries=20,
        is_max_cls=True,
        is_mhsa_float32=True,
        is_vit_pretrain=False,
        vit_layer_scale=True,
        decoder_layer_scale=True,
    ),
    loss_functions=dict(
        type=DiceCELoss, to_onehot_y=True, softmax=True, do_ds=True),
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
            project='synapse', name='swin-unetr-40k'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
