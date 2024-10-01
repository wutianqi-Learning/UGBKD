custom_hooks = [
    dict(
        after_epoch=True,
        after_iter=False,
        before_epoch=False,
        type='seg.engine.hooks.empty_cache_hook.EmptyCacheHook'),
    dict(
        eta_min=0.5,
        gamma=0.0016666666666666668,
        loss_names=[
            'loss_dsd3',
        ],
        type='seg.engine.hooks.schedule_hook.DistillLossWeightScheduleHookV2'),
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
        interval=20,
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
epoches = 300
find_unused_parameters = True
launcher = 'none'
load_from = 'work_dirs/multiscale_stage3_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96/5-run_20240522_194458/run0/best_Dice_75-56_epoch_300.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    architecture=dict(
        cfg_path=dict(
            backbone=dict(
                args=dict(channels=1, num_classes=17, s=1.0),
                classes=17,
                type='seg.models.nets.espnetv2.ESPNetv2Segmentation'),
            infer_cfg=dict(
                inf_size=[
                    96,
                    96,
                    96,
                ], infer_overlap=0.5, sw_batch_size=1),
            loss_functions=dict(
                softmax=True, to_onehot_y=True,
                type='monai.losses.DiceCELoss'),
            num_classes=14,
            roi_shapes=[
                96,
                96,
                96,
            ],
            type='seg.models.segmentors.monai_model.MonaiSeg'),
        pretrained=False),
    distiller=dict(
        distill_losses=dict(
            loss_dsd3=dict(
                cur_stage=3,
                in_chans=17,
                loss_weight=1.0,
                num_classes=17,
                num_stages=3,
                type='razor.models.losses.dsd.DSDLoss8')),
        loss_forward_mappings=dict(
            loss_dsd3=dict(
                feat_student=dict(from_student=True, recorder='logits'),
                label=dict(
                    data_idx=1, from_student=True, recorder='gt_labels'),
                logits_teacher=dict(from_student=False, recorder='logits'))),
        student_recorders=dict(
            feat1=dict(
                source='segmentor.backbone.bu_dec_l2',
                type=
                'mmrazor.models.task_modules.recorder.ModuleOutputsRecorder'),
            feat2=dict(
                source='segmentor.backbone.bu_dec_l3',
                type=
                'mmrazor.models.task_modules.recorder.ModuleOutputsRecorder'),
            gt_labels=dict(
                source='loss_functions',
                type='mmrazor.models.task_modules.recorder.ModuleInputsRecorder'
            ),
            logits=dict(
                source='segmentor',
                type=
                'mmrazor.models.task_modules.recorder.ModuleOutputsRecorder')),
        teacher_recorders=dict(
            logits=dict(
                source='segmentor',
                type=
                'mmrazor.models.task_modules.recorder.ModuleOutputsRecorder')),
        type='razor.models.distillers.ConfigurableDistiller'),
    teacher=dict(
        cfg_path=dict(
            backbone=dict(
                feature_size=48,
                img_size=[
                    96,
                    96,
                    96,
                ],
                in_channels=1,
                out_channels=17,
                spatial_dims=3,
                type='monai.networks.nets.SwinUNETR'),
            infer_cfg=dict(
                inf_size=[
                    96,
                    96,
                    96,
                ], infer_overlap=0.5, sw_batch_size=2),
            loss_functions=dict(
                softmax=True, to_onehot_y=True,
                type='monai.losses.DiceCELoss'),
            num_classes=17,
            roi_shapes=[
                96,
                96,
                96,
            ],
            type='seg.models.segmentors.monai_model.MonaiSeg'),
        pretrained=False),
    teacher_ckpt=
    'ckpts/swinunetr_base_1000e_word/best_Dice_84-89_epoch_1000.pth',
    type='razor.models.algorithms.SingleTeacherDistill')
num_classes = 17
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
student_model = dict(
    backbone=dict(
        args=dict(channels=1, num_classes=17, s=1.0),
        classes=17,
        type='seg.models.nets.espnetv2.ESPNetv2Segmentation'),
    infer_cfg=dict(
        inf_size=[
            96,
            96,
            96,
        ], infer_overlap=0.5, sw_batch_size=1),
    loss_functions=dict(
        softmax=True, to_onehot_y=True, type='monai.losses.DiceCELoss'),
    num_classes=14,
    roi_shapes=[
        96,
        96,
        96,
    ],
    type='seg.models.segmentors.monai_model.MonaiSeg')
teacher_ckpt = 'ckpts/swinunetr_base_1000e_word/best_Dice_84-89_epoch_1000.pth'
teacher_model = dict(
    backbone=dict(
        feature_size=48,
        img_size=[
            96,
            96,
            96,
        ],
        in_channels=1,
        out_channels=17,
        spatial_dims=3,
        type='monai.networks.nets.SwinUNETR'),
    infer_cfg=dict(
        inf_size=[
            96,
            96,
            96,
        ], infer_overlap=0.5, sw_batch_size=2),
    loss_functions=dict(
        softmax=True, to_onehot_y=True, type='monai.losses.DiceCELoss'),
    num_classes=17,
    roi_shapes=[
        96,
        96,
        96,
    ],
    type='seg.models.segmentors.monai_model.MonaiSeg')
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
        init_kwargs=dict(name='exp', project='mmsegmentation'),
        type='mmengine.visualization.vis_backend.WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmseg.visualization.local_visualizer.SegLocalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.vis_backend.LocalVisBackend'),
        dict(
            define_metric_cfg=dict(Dice='max'),
            init_kwargs=dict(name='exp', project='mmsegmentation'),
            type='mmengine.visualization.vis_backend.WandbVisBackend'),
    ])
work_dir = './save_dirs/multiscale_stage3_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96'
