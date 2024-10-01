from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.ofakd import OFALoss8
from razor.models.losses.kldiv_loss import KLDivergence3

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unetr.unetr_base_b2_pretrained_1000e_adamw_word_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetr_base_b2_pretrained_1000e_adamw_word_96x96x96/best_Dice_82-51_epoch_950.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_ofa1=dict(
                type=OFALoss8,
                in_chans=64,
                num_classes=17,
                num_stages=3,
                cur_stage=1,
                temperature=4.0,
                eps=1.2,
                loss_weight=1.0),
            loss_ofa2=dict(
                type=OFALoss8,
                in_chans=32,
                num_classes=17,
                num_stages=3,
                cur_stage=2,
                temperature=4.0,
                eps=1.2,
                loss_weight=1.0),
            loss_ofa3=dict(
                type=OFALoss8,
                in_chans=17,
                num_classes=17,
                num_stages=3,
                cur_stage=3,
                temperature=4.0,
                eps=1.2,
                loss_weight=1.0),
        ),
        student_recorders=dict(
            feat1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1'),
            feat2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_ofa1=dict(
                feat_student=dict(from_student=True, recorder='feat1'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)),
            loss_ofa2=dict(
                feat_student=dict(from_student=True, recorder='feat2'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)),
            loss_ofa3=dict(
                feat_student=dict(from_student=True, recorder='logits'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)),
        )))

find_unused_parameters = True
