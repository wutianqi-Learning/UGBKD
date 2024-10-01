# only at6
# residual to conv
from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.exkd_loss import EXKDV2_Loss

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_300e_sgd_word_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_300e_sgd_word_96x96x96/best_Dice_84-12_epoch_300.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            # loss_at1=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=32,
            #     teacher_channels=64,
            #     loss_weight=1.0),
            # loss_at2=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=64,
            #     teacher_channels=128,
            #     loss_weight=1.0),
            loss_at3=dict(
                type=EXKDV2_Loss,
                spatial_dims=3,
                student_channels=128,
                teacher_channels=256,
                loss_weight=1.0),
            # loss_at4=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=256,
            #     teacher_channels=512,
            #     loss_weight=1.0),
            # loss_at5=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=64,
            #     teacher_channels=128,
            #     loss_weight=1.0),
            # loss_at6=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=32,
            #     teacher_channels=64,
            #     loss_weight=1.0),
            # loss_at7=dict(
            #     type=EXKDV2_Loss,
            #     spatial_dims=3,
            #     student_channels=14,
            #     teacher_channels=14,
            #     loss_weight=1.0),
        ),
        student_recorders=dict(
            # down_conv1=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.down_layer1.conv'),
            # down_conv2=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.down_layer2.conv'),
            down_conv3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer3.conv'),
            # bottom_conv=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.bottom_layer.conv'),
            # up_conv1=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.up_layer1.conv'),
            # up_conv2=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.up_layer2.conv'),
            # up_conv3=dict(
            #     type=ModuleOutputsRecorder,
            #     source='segmentor.backbone.up_layer3.conv')
        ),
        teacher_recorders=dict(
            # down1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1'),
            # down_ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1.conv'),
            # down2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer2'),
            # down_ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer2.conv'),
            down3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3'),
            down_ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3.conv'),
            # bottom=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer'),
            # bottom_ru=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer.conv'),
            # up1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1.1'),
            # up_ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1.1.conv'),
            # up2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1'),
            # up_ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1.conv'),
            # up3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer3.1'),
            # up_ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer3.1.conv')
        ),
        loss_forward_mappings=dict(
            # loss_at1=dict(
            #     s_feature=dict(from_student=True, recorder='down_conv1'),
            #     t_feature=dict(from_student=False, recorder='down1'),
            #     t_residual=dict(from_student=False, recorder='down_ru1')),
            # loss_at2=dict(
            #     s_feature=dict(from_student=True, recorder='down_conv2'),
            #     t_feature=dict(from_student=False, recorder='down2'),
            #     t_residual=dict(from_student=False, recorder='down_ru2')),
            loss_at3=dict(
                s_feature=dict(from_student=True, recorder='down_conv3'),
                t_feature=dict(from_student=False, recorder='down3'),
                t_residual=dict(from_student=False, recorder='down_ru3')),
            # loss_at4=dict(
            #     s_feature=dict(from_student=True, recorder='bottom_conv'),
            #     t_feature=dict(from_student=False, recorder='bottom'),
            #     t_residual=dict(from_student=False, recorder='bottom_ru')),
            # loss_at5=dict(
            #     s_feature=dict(from_student=True, recorder='up_conv1'),
            #     t_feature=dict(from_student=False, recorder='up1'),
            #     t_residual=dict(from_student=False, recorder='up_ru1')),
            # loss_at6=dict(
            #     s_feature=dict(from_student=True, recorder='up_conv2'),
            #     t_feature=dict(from_student=False, recorder='up2'),
            #     t_residual=dict(from_student=False, recorder='up_ru2')),
            # loss_at7=dict(
            #     s_feature=dict(from_student=True, recorder='up_conv3'),
            #     t_feature=dict(from_student=False, recorder='up3'),
            #     t_residual=dict(from_student=False, recorder='up_ru3')),
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
