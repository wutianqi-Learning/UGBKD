from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.cwd import ChannelWiseDivergenceWithU
from razor.models.losses.dsd import DSDLoss8, DKDLoss8
from seg.engine.hooks.get_epoc_student_hook import SetEpochInfoHook
from seg.engine.hooks.schedule_hook import DistillLossWeightScheduleHookV2

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...unet.unetmod_tiny_d8_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa


total_epochs = 1000
num_classes = 14
teacher_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            loss_cwd=dict(
                type=ChannelWiseDivergenceWithU,
                dae_network=teacher_model,
                dae_ckpt=teacher_ckpt,
                epoch=0,
                gamma=2.0,
                consistency=1.0,
                consistency_rampup=0.0,
                tau=3,
                loss_weight=1.0,
            ),
            loss_dsd3=dict(
                type=DKDLoss8,
                in_chans=num_classes,
                num_classes=num_classes,
                num_stages=3,
                cur_stage=3,
                loss_weight=1.0,
            )),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
            ),
            loss_dsd3=dict(
                feat_student=dict(from_student=True, recorder='logits'),
                logits_teacher=dict(from_student=False, recorder='logits'),
                label=dict(recorder='gt_labels', from_student=True, data_idx=1),
            ),
        )))

find_unused_parameters = True
custom_hooks = [dict(type=SetEpochInfoHook)]
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='cwd-unet-base-unet-tiny-1000e'),
        )
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
custom_hooks.append(
    dict(
        type=DistillLossWeightScheduleHookV2,
        loss_names=['loss_dsd3'],
        eta_min=0.5, gamma=0.5/total_epochs
    ))