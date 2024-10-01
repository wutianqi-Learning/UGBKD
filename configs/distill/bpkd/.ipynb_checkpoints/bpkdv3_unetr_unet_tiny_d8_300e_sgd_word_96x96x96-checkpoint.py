from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.bpkd import BPKDV3


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
            loss_bpkd=dict(
                type=BPKDV3,
                num_classes=17,
                # alpha=0.1,
                edge_weight=0.5,
                body_weight=0.2
            )),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_bpkd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
                gt_labels=dict(
                    recorder='gt_labels', from_student=True, data_idx=1)
            ))))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='dist-swinunetr-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
