from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.architectures.connectors import ConvModuleConnector
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from razor.models.losses.emkd_losses import IMD
from razor.models.losses.exkd_loss import Attention_Loss
from mmrazor.models.losses import L2Loss

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
                type=Attention_Loss,
                spatial_dims=3,
                student_channels=32,
                teacher_channels=64,
                loss_weight=1.0),
            # loss_at2=dict(
            #     type=IMD,
            #     loss_weight=1.0),
            loss_at3=dict(
                type=Attention_Loss,
                spatial_dims=3,
                student_channels=128,
                teacher_channels=256,
                loss_weight=1.0),
            # loss_at4=dict(
            #     type=IMD,
            #     loss_weight=1.0),
            # loss_at5=dict(
            #     type=IMD,
            #     loss_weight=1.0),
            # loss_at6=dict(
            #     type=IMD,
            #     loss_weight=1.0),
        ),
        student_recorders=dict(
            down_conv1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1.conv'),
            # down_conv2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer2.conv'),
            down_conv3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3.conv'),
            # bottom_conv=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer.conv'),
            # up_conv1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1.conv'),
            # up_conv2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.conv'),
        ),
        teacher_recorders=dict(
            down1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1'),
            # down2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer2'),
            down3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3'),
            # bottom=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer'),
            # up1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1.1'),
            # up2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1'),
        ),
        # connectors=dict(
        #     loss_s1_sfeat=dict(type=ConvModuleConnector,
        #                        conv_cfg=dict(type='Conv3d'),
        #                        in_channel=32,
        #                        out_channel=64),
        #     loss_s3_sfeat=dict(type=ConvModuleConnector,
        #                        conv_cfg=dict(type='Conv3d'),
        #                        in_channel=128,
        #                        out_channel=256),
        # ),
        loss_forward_mappings=dict(
            loss_at1=dict(
                s_feature=dict(from_student=True, recorder='down_conv1',
                               # connector='loss_s1_sfeat'
                               ),
                t_feature=dict(from_student=False, recorder='down1')),
            # loss_at2=dict(
            #     s_feature=dict(from_student=True, recorder='down_conv2'),
            #     t_feature=dict(from_student=False, recorder='down2')),
            loss_at3=dict(
                s_feature=dict(from_student=True, recorder='down_conv3',
                               # connector='loss_s3_sfeat'
                               ),
                t_feature=dict(from_student=False, recorder='down3')),
            # loss_at4=dict(
            #     s_feature=dict(from_student=True, recorder='bottom_conv'),
            #     t_feature=dict(from_student=False, recorder='bottom')),
            # loss_at5=dict(
            #     s_feature=dict(from_student=True, recorder='up_conv1'),
            #     t_feature=dict(from_student=False, recorder='up1')),
            # loss_at6=dict(
            #     s_feature=dict(from_student=True, recorder='up_conv2'),
            #     t_feature=dict(from_student=False, recorder='up2')),
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
