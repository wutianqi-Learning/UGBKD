from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, ModuleInputsRecorder
from razor.models.losses.dsd import DSDLoss8
from seg.engine.hooks.schedule_hook import DistillLossWeightScheduleHookV2

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_3000e_adamw import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...swin_unetr.swinunetr_base_5000e_synapse import model as teacher_model  # noqa
    from ...swin_unetr.swinunetr_tiny_b1_3000e_adamw_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'  # noqa: E501
model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        distill_losses=dict(
            # loss_dsd1=dict(
            #     type=DSDLoss8,
            #     in_chans=96,
            #     num_classes=14,
            #     num_stages=5,
            #     cur_stage=1,
            #     loss_weight=1.0,
            # ),
            # loss_dsd2=dict(
            #     type=DSDLoss8,
            #     in_chans=48,
            #     num_classes=14,
            #     num_stages=5,
            #     cur_stage=2,
            #     loss_weight=1.0,
            # ),
            # loss_dsd3=dict(
            #     type=DSDLoss8,
            #     in_chans=24,
            #     num_classes=14,
            #     num_stages=5,
            #     cur_stage=3,
            #     loss_weight=1.0,
            # ),
            # loss_dsd4=dict(
            #     type=DSDLoss8,
            #     in_chans=12,
            #     num_classes=14,
            #     num_stages=5,
            #     cur_stage=4,
            #     loss_weight=1.0,
            # ),
            # loss_dsd5=dict(
            #     type=DSDLoss8,
            #     in_chans=12,
            #     num_classes=14,
            #     num_stages=5,
            #     cur_stage=5,
            #     loss_weight=1.0,
            # ),
            loss_dsd6=dict(
                type=DSDLoss8,
                in_chans=14,
                num_classes=14,
                num_stages=5,
                cur_stage=5,
                loss_weight=1.0,
            ),
        ),
        student_recorders=dict(
            # feat1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder5'),
            # feat2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder4'),
            # feat3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder3'),
            # feat4=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder2'),
            # feat5=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder1'),
            logits=dict(type=ModuleOutputsRecorder, source='segmentor'),
            gt_labels=dict(type=ModuleInputsRecorder, source='loss_functions')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='segmentor')),
        loss_forward_mappings=dict(
            # loss_dsd1=dict(
            #     feat_student=dict(from_student=True, recorder='feat1'),
            #     logits_teacher=dict(from_student=False, recorder='logits'),
            #     label=dict(
            #         recorder='gt_labels', from_student=True, data_idx=1),
            # ),
            # loss_dsd2=dict(
            #     feat_student=dict(from_student=True, recorder='feat2'),
            #     logits_teacher=dict(from_student=False, recorder='logits'),
            #     label=dict(
            #         recorder='gt_labels', from_student=True, data_idx=1),
            # ),
            # loss_dsd3=dict(
            #     feat_student=dict(from_student=True, recorder='feat3'),
            #     logits_teacher=dict(from_student=False, recorder='logits'),
            #     label=dict(
            #         recorder='gt_labels', from_student=True, data_idx=1),
            # ),
            # loss_dsd4=dict(
            #     feat_student=dict(from_student=True, recorder='feat4'),
            #     logits_teacher=dict(from_student=False, recorder='logits'),
            #     label=dict(
            #         recorder='gt_labels', from_student=True, data_idx=1),
            # ),
            # loss_dsd5=dict(
            #     feat_student=dict(from_student=True, recorder='feat5'),
            #     logits_teacher=dict(from_student=False, recorder='logits'),
            #     label=dict(
            #         recorder='gt_labels', from_student=True, data_idx=1),
            # ),
            loss_dsd6=dict(
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
        loss_names=[
            # 'loss_dsd1',
            # 'loss_dsd2',
            # 'loss_dsd3',
            # 'loss_dsd4',
            # 'loss_dsd5',
            'loss_dsd6'],
        eta_min=0.5, gamma=0.5/1000
    ))