from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from razor.models.losses.mskd_cakd import MSKD_CAKD
from razor.models.losses.kldiv_loss import CriterionKD

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...uxnet.uxnet_b1_2000e_adamw_noscheduler_word_96x96x96 import model as teacher_model  # noqa
    from ...espnetv2.espnetv2_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/uxnet_b1_2000e_adamw_noscheduler_word_96x96x96/best_Dice_85-29_epoch_1900.pth'  # noqa: E501

class_num = 17
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            # up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l2'),
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l3'),#24
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l4'),#48
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor')#96
        ),
        teacher_recorders=dict(
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder2'),#24
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder1'),#48
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor')#96
        ),
        distill_losses=dict(
            loss_cakd1=dict(
                type=MSKD_CAKD,
                # interpolate=True,
                s_in_chans=32,
                t_in_chans=96,
                num_classes=class_num,
                loss_weight=4),
            loss_cakd2=dict(
                type=MSKD_CAKD,
                # interpolate=True,
                s_in_chans=class_num,
                t_in_chans=48,
                num_classes=class_num,
                loss_weight=4),
            loss_cakd3=dict(
                type=MSKD_CAKD,
                s_in_chans=class_num,
                t_in_chans=class_num,
                num_classes=class_num,
                loss_weight=8),
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
