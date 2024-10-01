from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from mmseg.models.losses.kldiv_loss import KLDivLoss
from mmrazor.models.losses import KLDivergence
from razor.models.losses.kldiv_loss import KLDivergence3, CriterionKD, CriterionKDWithU
from seg.engine.hooks.get_epoc_student_hook import SetEpochInfoHook

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
            loss_kl=dict(
                type=CriterionKDWithU,
                tau=4,
                loss_weight=10.0,
                dae_ckpt="/home/jz207/workspace/zhangdw/monai_mmengine/ckpts/dea_segresnet_1000e_sgd_syanpse_96x96x96/best_Dice_78-50_epoch_1000.pth",
                out_channels=14,
                epoch=0,
                consistency=0.1,
                consistency_rampup=500.0
                # reduction='mean'
            )),
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))

find_unused_parameters = True
custom_hooks = [dict(type=SetEpochInfoHook)]
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
