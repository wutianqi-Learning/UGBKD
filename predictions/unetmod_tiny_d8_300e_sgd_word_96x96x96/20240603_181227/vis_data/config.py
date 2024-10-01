custom_hooks = [
    dict(
        after_epoch=True,
        after_iter=False,
        before_epoch=False,
        type='seg.engine.hooks.empty_cache_hook.EmptyCacheHook'),
]
dataloader_cfg = dict(
    RandFlipd_prob=0.2,
    RandRotate90d_prob=0.2,
    RandScaleIntensityd_prob=0.1,
    RandShiftIntensityd_prob=0.1,
    a_max=250,
    a_min=-175.0,
    b_max=1.0,
    b_min=0.0,
    batch_size=1,
    data_dir='data/WORD',
    data_name='WORD',
    distributed=False,
    json_list='dataset.json',
    meta_info='seg.datasets.monai_dataset.WORD_METAINFO',
    num_samples=4,
    roi_x=96,
    roi_y=96,
    roi_z=96,
    space_x=1.5,
    space_y=1.5,
    space_z=2.0,
    train_case_nums=100,
    use_normal_dataset=True,
    use_smart_dataset=False,
    use_test_data=False,
    workers=2)
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=25,
        max_keep_ckpts=1,
        rule='greater',
        save_best=[
            'Dice',
        ],
        type='seg.engine.hooks.MyCheckpointHook'),
    logger=dict(
        interval=10,
        type='seg.engine.hooks.logger_hook.MyLoggerHook',
        val_interval=1),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(type='mmseg.engine.hooks.SegVisualizationHook'))
default_scope = None
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'ckpts/unetmod_tiny_d8_300e_sgd_word_96x96x96/best_Dice_76-40_epoch_300.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    backbone=dict(
        channels=(
            32,
            64,
            128,
            256,
        ),
        in_channels=1,
        num_res_units=0,
        out_channels=17,
        spatial_dims=3,
        strides=(
            2,
            2,
            2,
        ),
        type='seg.models.unet.monai_unet_mod.UNetMod'),
    infer_cfg=dict(
        inf_size=[
            96,
            96,
            96,
        ], infer_overlap=0.5, sw_batch_size=4),
    loss_functions=dict(
        softmax=True, to_onehot_y=True, type='monai.losses.DiceCELoss'),
    num_classes=17,
    roi_shapes=[
        96,
        96,
        96,
    ],
    type='seg.models.segmentors.monai_model.MonaiSeg')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        lr=0.01, momentum=0.9, type='torch.optim.SGD', weight_decay=0.0005),
    type='mmengine.optim.OptimWrapper')
optimizer = dict(
    lr=0.01, momentum=0.9, type='torch.optim.SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=300,
        eta_min=1e-06,
        power=0.9,
        type='mmengine.optim.scheduler.PolyLR'),
]
resume = False
roi = [
    96,
    96,
    96,
]
runner_type = 'seg.engine.runner.monai_runner.MonaiRunner'
save = True
test_cfg = dict(type='seg.engine.runner.monai_loops.MonaiTestLoop')
test_evaluator = dict(
    metrics=dict(
        metrics=[
            'Dice',
            'HD95',
        ],
        num_classes=17,
        type='seg.evaluation.metrics.monai_metric.MonaiMetric'),
    type='seg.evaluation.monai_evaluator.MonaiEvaluator')
test_mode = True
train_cfg = dict(
    max_epochs=300,
    type='mmengine.runner.loops.EpochBasedTrainLoop',
    val_begin=100,
    val_interval=25)
val_cfg = dict(type='seg.engine.runner.monai_loops.MonaiValLoop')
val_evaluator = dict(
    metrics=dict(
        metrics=[
            'Dice',
            'HD95',
        ],
        num_classes=17,
        type='seg.evaluation.metrics.monai_metric.MonaiMetric'),
    type='seg.evaluation.monai_evaluator.MonaiEvaluator')
vis_backends = [
    dict(type='mmengine.visualization.vis_backend.LocalVisBackend'),
    dict(
        define_metric_cfg=dict(Dice='max'),
        init_kwargs=dict(name='unet-tiny-sgd-300e', project='word'),
        type='mmengine.visualization.vis_backend.WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmseg.visualization.local_visualizer.SegLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.vis_backend.LocalVisBackend'),
        dict(
            define_metric_cfg=dict(Dice='max'),
            init_kwargs=dict(name='unet-tiny-sgd-300e', project='word'),
            type='mmengine.visualization.vis_backend.WandbVisBackend'),
    ])
work_dir = './save_dirs/unetmod_tiny_d8_300e_sgd_word_96x96x96'
