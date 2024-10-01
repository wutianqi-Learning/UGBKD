# only at6
# residual to conv
from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.boundary_loss import BoundaryKDV1
from razor.models.losses.hd_loss import LogHausdorffDTLoss
from seg.engine.hooks.schedule_hook import DistillLossWeightScheduleHook

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_1000e_word import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/swinunetr_base_1000e_word/best_Dice_84-89_epoch_1000.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_boundary=dict(
                type=BoundaryKDV1,
                loss_weight=1.0,
            ),
            loss_hd=dict(
                type=LogHausdorffDTLoss,
                loss_weight=0.,
            ),
        ),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')
        ),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')
        ),
        loss_forward_mappings=dict(
            loss_boundary=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ),
            loss_hd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                target=dict(
                    recorder='gt_labels', from_student=True, data_idx=1),
            ))
        ))

custom_hooks.append(
    dict(
        type=DistillLossWeightScheduleHook,
        eta_min=0.5, gamma=0.5/1000
    ))

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
