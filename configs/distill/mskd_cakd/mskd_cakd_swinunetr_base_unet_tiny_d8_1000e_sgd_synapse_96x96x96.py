from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from razor.models.losses.mskd_cakd import MSKD_CAKD

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_5000e_synapse import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1'),
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor')
        ),
        teacher_recorders=dict(
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder3'),
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder2'),
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor')
        ),
        distill_losses=dict(
            loss_cakd1=dict(
                type=MSKD_CAKD,
                s_in_chans=64,
                t_in_chans=96,
                num_classes=14,
                loss_weight=0.4),
            loss_cakd2=dict(
                type=MSKD_CAKD,
                s_in_chans=32,
                t_in_chans=48,
                num_classes=14,
                loss_weight=0.4),
            loss_cakd3=dict(
                type=MSKD_CAKD,
                s_in_chans=14,
                t_in_chans=14,
                num_classes=14,
                loss_weight=0.8),
        ),
        loss_forward_mappings=dict(
            loss_cakd1=dict(
                student_outputs=dict(from_student=True, recorder='up_layer1'),
                teacher_outputs=dict(from_student=False, recorder='up_layer1'),
            ),
            loss_cakd2=dict(
                student_outputs=dict(from_student=True, recorder='up_layer2'),
                teacher_outputs=dict(from_student=False, recorder='up_layer2'),
            ),
            loss_cakd3=dict(
                student_outputs=dict(from_student=True, recorder='up_layer3'),
                teacher_outputs=dict(from_student=False, recorder='up_layer3'),
            ),
        )))

find_unused_parameters = True
