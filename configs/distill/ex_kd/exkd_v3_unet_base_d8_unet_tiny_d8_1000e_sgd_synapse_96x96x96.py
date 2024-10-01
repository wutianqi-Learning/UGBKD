from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.exkd_loss import EXKDV2_Loss
# from razor.models.architectures.connectors.exkd_connector import R2AConvertor
# from seg.models.utils.ex_kd import R2AModule

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_at1=dict(
                type=EXKDV2_Loss,
                spatial_dims=3,
                alpha=1.0,
                beta=0.5,
                student_channels=32,
                teacher_channels=64,
                loss_weight=1.0),
            loss_at3=dict(
                type=EXKDV2_Loss,
                spatial_dims=3,
                alpha=1.0,
                beta=0.5,
                student_channels=128,
                teacher_channels=256,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            down_conv1=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer1.conv'),
            down_conv3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer3.conv'),
        ),
        teacher_recorders=dict(
            down1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1'),
            down_ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1.residual'),
            down3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3'),
            down_ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3.residual'),
        ),
        loss_forward_mappings=dict(
            loss_at1=dict(
                s_feature=dict(from_student=True, recorder='down_conv1'),
                t_feature=dict(from_student=False, recorder='down1'),
                t_residual=dict(from_student=False, recorder='down_ru1')),
            loss_at3=dict(
                s_feature=dict(from_student=True, recorder='down_conv3'),
                t_feature=dict(from_student=False, recorder='down3'),
                t_residual=dict(from_student=False, recorder='down_ru3')),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='kd-unet-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
