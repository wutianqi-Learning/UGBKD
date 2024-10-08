from torch.optim import SGD
from mmengine.optim.scheduler import PolyLR
from mmengine.optim import OptimWrapper
from mmengine.hooks import IterTimerHook, ParamSchedulerHook, DistSamplerSeedHook
from seg.engine.hooks.empty_cache_hook import EmptyCacheHook
from mmseg.engine.hooks import SegVisualizationHook
from seg.engine.hooks import MyCheckpointHook
from seg.engine.hooks.logger_hook import MyLoggerHook
from mmengine.runner.loops import EpochBasedTrainLoop
from seg.engine.runner.monai_loops import MonaiValLoop, MonaiTestLoop

train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=300, val_begin=100, val_interval=25)
val_cfg = dict(type=MonaiValLoop)
test_cfg = dict(type=MonaiTestLoop)
# optimizer
optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
# optimizer = dict(type=SGD, lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-6,
        power=0.9,
        begin=0,
        end=300,
        by_epoch=True)
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=MyLoggerHook, interval=10, val_interval=1),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=True,
                    interval=25,
                    max_keep_ckpts=1,
                    save_best=['Dice'], rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

custom_hooks = [
    dict(
        type=EmptyCacheHook,
        before_epoch=False,
        after_epoch=True,
        after_iter=False)]
