from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmrazor.models.losses import ATLoss, L2Loss
from razor.models.architectures.connectors.exkd_connector import R2AConvertor
from seg.models.utils.ex_kd import R2AModule

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...unet_kd.unetmod_kd_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

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
                type=ATLoss,
                loss_weight=2000.0),
            loss_at2=dict(
                type=ATLoss,
                loss_weight=2000.0),
            loss_at3=dict(
                type=ATLoss,
                loss_weight=2000.0),
            loss_at4=dict(
                type=ATLoss,
                loss_weight=2000.0),
            loss_at5=dict(
                type=ATLoss,
                loss_weight=2000.0),
            loss_at6=dict(
                type=ATLoss,
                loss_weight=2000.0),
            loss_at7=dict(
                type=ATLoss,
                loss_weight=2000.0),
        ),
        student_recorders=dict(
            down_conv1=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer1.attn.attn'),
            down_conv2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer2.attn.attn'),
            down_conv3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.down_layer3.attn.attn'),
            bottom_conv=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.bottom_layer.attn.attn'),
            up_conv1=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer1.attn.attn'),
            up_conv2=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer2.attn.attn'),
            up_conv3=dict(
                type=ModuleOutputsRecorder,
                source='segmentor.backbone.up_layer3.attn.attn')),
        teacher_recorders=dict(
            down_ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer1.conv'),
            down_ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer2.conv'),
            down_ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.down_layer3.conv'),
            bottom_ru=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bottom_layer.conv'),
            up_ru1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1.1.conv'),
            up_ru2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2.1.conv'),
            up_ru3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer3.1.conv')),
        connectors=dict(
            loss_s1_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=32,
                               teacher_channels=64),
            loss_s2_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=64,
                               teacher_channels=128),
            loss_s3_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=128,
                               teacher_channels=256),
            loss_s4_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=256,
                               teacher_channels=512),
            loss_s5_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=64,
                               teacher_channels=128),
            loss_s6_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=32,
                               teacher_channels=64),
            loss_s7_sfeat=dict(type=R2AConvertor,
                               spatial_dims=3,
                               student_channels=14,
                               teacher_channels=14),
        ),
        loss_forward_mappings=dict(
            loss_at1=dict(
                s_feature=dict(from_student=True, recorder='down_conv1', connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='down_ru1')),
            loss_at2=dict(
                s_feature=dict(from_student=True, recorder='down_conv2', connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='down_ru2')),
            loss_at3=dict(
                s_feature=dict(from_student=True, recorder='down_conv3', connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='down_ru3')),
            loss_at4=dict(
                s_feature=dict(from_student=True, recorder='bottom_conv', connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bottom_ru')),
            loss_at5=dict(
                s_feature=dict(from_student=True, recorder='up_conv1', connector='loss_s5_sfeat'),
                t_feature=dict(from_student=False, recorder='up_ru1')),
            loss_at6=dict(
                s_feature=dict(from_student=True, recorder='up_conv2', connector='loss_s6_sfeat'),
                t_feature=dict(from_student=False, recorder='up_ru2')),
            loss_at7=dict(
                s_feature=dict(from_student=True, recorder='up_conv3', connector='loss_s7_sfeat'),
                t_feature=dict(from_student=False, recorder='up_ru3')),
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
