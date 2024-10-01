from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import EXDistiller
from razor.models.architectures.connectors import EXKDConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from mmrazor.models.losses import L2Loss
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unet_base_sgd_synapse import model as teacher_model  # noqa
    from ...unet.noskip_unet_kd_tiny_sgd_synapse import model as student_model  # noqa

teacher_ckpt = 'ckpts/unet50_sgd_synapse/best_Dice_83-29_epoch_780.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=EXDistiller,
        student_recorders=dict(
            ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.0.1'),
            ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.0.1'),
            ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.0.1'),
            ru4=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.1.submodule.1'),
            ru5=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.2.1'),
            ru6=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.2.1'),
            ru7=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.2.1')),
        teacher_recorders=dict(
            ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.0'),
            ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.0'),
            ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.0'),
            ru4=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.1.submodule'),
            ru5=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.1.submodule.2.1'),
            ru6=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.1.submodule.2.1'),
            ru7=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.model.2.1')),
        distill_losses=dict(
            loss_1=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_2=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_3=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_4=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_5=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_6=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
            loss_7=dict(type=L2Loss, normalize=True, loss_weight=0.5, dist=True),
        ),
        connectors=dict(
            loss_s1_sfeat=dict(type=EXKDConnector,
                               spatial_dim=3,
                               student_channels=32,
                               teacher_channels=64),
            loss_s2_sfeat=dict(type=EXKDConnector,
                               spatial_dim=3,
                               student_channels=64,
                               teacher_channels=128),
            loss_s3_sfeat=dict(type=EXKDConnector,
                               spatial_dim=3,
                               student_channels=128,
                               teacher_channels=256),
            loss_s4_sfeat=dict(type=EXKDConnector,
                               spatial_dim=3,
                               student_channels=256,
                               teacher_channels=512),
            # loss_s5_sfeat=dict(type=EXKDConnector,
            #                    student_channels=128,
            #                    teacher_channels=128),
            # loss_s6_sfeat=dict(type=EXKDConnector,
            #                    student_channels=64,
            #                    teacher_channels=64),
            # loss_s7_sfeat=dict(type=EXKDConnector,
            #                    student_channels=14,
            #                    teacher_channels=14),
        ),
        loss_forward_mappings=dict(
            loss_1=dict(
                s_feature=dict(
                    recorder='ru1',
                    from_student=True,
                    connector='loss_s1_sfeat'
                ),
                t_feature=dict(recorder='ru1', from_student=False)),
            loss_2=dict(
                s_feature=dict(
                    recorder='ru2',
                    from_student=True,
                    connector='loss_s2_sfeat'
                ),
                t_feature=dict(recorder='ru2', from_student=False)),
            loss_3=dict(
                s_feature=dict(
                    recorder='ru3',
                    from_student=True,
                    connector='loss_s3_sfeat'
                ),
                t_feature=dict(recorder='ru3', from_student=False)),
            loss_4=dict(
                s_feature=dict(
                    recorder='ru4',
                    from_student=True,
                    connector='loss_s4_sfeat'
                ),
                t_feature=dict(recorder='ru4', from_student=False)),
            loss_5=dict(
                s_feature=dict(
                    recorder='ru5',
                    from_student=True,
                    # connector='loss_s2_sfeat'
                ),
                t_feature=dict(recorder='ru5', from_student=False)),
            loss_6=dict(
                s_feature=dict(
                    recorder='ru6',
                    from_student=True,
                    # connector='loss_s2_sfeat'
                ),
                t_feature=dict(recorder='ru6', from_student=False)),
            loss_7=dict(
                s_feature=dict(
                    recorder='ru7',
                    from_student=True,
                    # connector='loss_s2_sfeat'
                ),
                t_feature=dict(recorder='ru7', from_student=False)),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='exkd_fcn_r50_fcn_r18-80k'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
