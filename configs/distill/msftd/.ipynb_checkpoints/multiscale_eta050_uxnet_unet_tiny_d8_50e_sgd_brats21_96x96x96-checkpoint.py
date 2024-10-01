from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.dsd import DSDLoss8
from seg.engine.hooks.schedule_hook import DistillLossWeightScheduleHookV2

with read_base():
    from ..._base_.datasets.brats21 import *  # noqa
    from ..._base_.schedules.schedule_50e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...uxnet.uxnet_b2_300e_adamw_noscheduler_brats21_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_50e_sgd_brats21_96x96x96 import model as student_model  # noqa

dataloader_cfg.update(
    dict(batch_size=2)
)

student_model.update(
    dict(
        infer_cfg=dict(
            sw_batch_size=2,    # number of sliding window batch size
        )))

teacher_ckpt = 'ckpts/uxnet_b2_300e_adamw_noscheduler_brats21_96x96x96/best_Dice_90-32_epoch_175.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_dsd1=dict(
                type=DSDLoss8,
                in_chans=64,
                num_classes=3,
                num_stages=3,
                cur_stage=1,
                bd_include_background=True,
                hd_include_background=True,
                one_hot_target=False,
                sigmoid=True,
                softmax=False,
                loss_weight=1.0,
            ),
            loss_dsd2=dict(
                type=DSDLoss8,
                in_chans=32,
                num_classes=3,
                num_stages=3,
                cur_stage=2,
                bd_include_background=True,
                hd_include_background=True,
                one_hot_target=False,
                sigmoid=True,
                softmax=False,
                loss_weight=1.0,
            ),
            loss_dsd3=dict(
                type=DSDLoss8,
                in_chans=3,
                num_classes=3,
                num_stages=3,
                cur_stage=3,
                bd_include_background=True,
                hd_include_background=True,
                one_hot_target=False,
                sigmoid=True,
                softmax=False,
                loss_weight=1.0,
            )
        ),
        student_recorders=dict(
            feat1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1'),
            feat2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_dsd1=dict(
                feat_student=dict(from_student=True, recorder='feat1'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
            loss_dsd2=dict(
                feat_student=dict(from_student=True, recorder='feat2'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
            loss_dsd3=dict(
                feat_student=dict(from_student=True, recorder='logits'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
        )))

find_unused_parameters = True

custom_hooks.append(
    dict(
        type=DistillLossWeightScheduleHookV2,
        loss_names=['loss_dsd1', 'loss_dsd2', 'loss_dsd3'],
        eta_min=0.5, gamma=0.5/1000
    ))