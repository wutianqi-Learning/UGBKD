from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.dsd import DSDLoss8_BCD

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_1000e_word import model as teacher_model  # noqa
    from ...espnetv2.espnetv2_300e_sgd_word_96x96x96 import model as student_model  # noqa

num_classes = 17
teacher_ckpt = 'ckpts/swinunetr_base_1000e_word/best_Dice_84-89_epoch_1000.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_dsd3=dict(
                type=DSDLoss8_BCD,
                in_chans=num_classes,
                num_classes=num_classes,
                num_stages=3,
                cur_stage=3,
                loss_weight=1.0,
            )
        ),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_dsd3=dict(
                feat_student=dict(from_student=True, recorder='logits'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
        )))

find_unused_parameters = True
