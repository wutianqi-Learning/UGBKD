from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder, \
    ModuleInputsRecorder
from razor.models.architectures.connectors.conv3d_connector import Conv3DConnector
from razor.models.losses.mskd_cakd import MSKDCAKDLoss, CAKD, FNKD

with read_base():
    from ..._base_.datasets.synapse import *  # noqa
    from ..._base_.schedules.schedule_1000e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...unet.unetmod_base_d8_1000e_sgd_synapse_96x96x96 import model as teacher_model  # noqa
    from ...lcovnet.lcovnet_ds_1000e_sgd_synapse_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            decoder_block2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.Decode_block2'),
            decoder_block3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.Decode_block3'),
            decoder_block4=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.Decode_block4'),
            segout2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.segout2'),
            segout3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.segout3'),
            segout4=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.segout4')
        ),
        teacher_recorders=dict(
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer1'),
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer2'),
            up_layer3_in=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.up_layer3')
        ),
        distill_losses=dict(
            loss_cakd1=dict(type=CAKD, loss_weight=0.2),
            loss_cakd2=dict(type=CAKD, loss_weight=0.2),
            loss_cakd3=dict(type=CAKD, loss_weight=0.4),
            loss_fnkd1=dict(type=FNKD, loss_weight=0.4),
            loss_fnkd2=dict(type=FNKD, loss_weight=0.4),
            loss_fnkd3=dict(type=FNKD, loss_weight=0.8),
        ),
        connectors=dict(
            pred_t_1=dict(
                type=Conv3DConnector,
                args=dict(in_channels=128,
                          out_channels=14,
                          kernel_size=1,
                          padding=0)),
            pred_t_2=dict(
                type=Conv3DConnector,
                args=dict(in_channels=64,
                          out_channels=14,
                          kernel_size=1,
                          padding=0)),
        ),
        loss_forward_mappings=dict(
            loss_cakd1=dict(
                student_outputs=dict(from_student=True, recorder='segout2'),
                teacher_outputs=dict(from_student=False, recorder='up_layer1', connector='pred_t_1'),
            ),
            loss_cakd2=dict(
                student_outputs=dict(from_student=True, recorder='segout3'),
                teacher_outputs=dict(from_student=False, recorder='up_layer2', connector='pred_t_2'),
            ),
            loss_cakd3=dict(
                student_outputs=dict(from_student=True, recorder='segout4'),
                teacher_outputs=dict(from_student=False, recorder='up_layer3'),
            ),
            loss_fnkd1=dict(
                student_outputs=dict(from_student=True, recorder='segout2'),
                teacher_outputs=dict(from_student=False, recorder='up_layer1', connector='pred_t_1'),
                student_feature=dict(from_student=True, recorder='decoder_block2'),
                teacher_feature=dict(from_student=False, recorder='up_layer1'),
            ),
            loss_fnkd2=dict(
                student_outputs=dict(from_student=True, recorder='segout3'),
                teacher_outputs=dict(from_student=False, recorder='up_layer2', connector='pred_t_2'),
                student_feature=dict(from_student=True, recorder='decoder_block3'),
                teacher_feature=dict(from_student=False, recorder='up_layer2'),
            ),
            loss_fnkd3=dict(
                student_outputs=dict(from_student=True, recorder='segout4'),
                teacher_outputs=dict(from_student=False, recorder='up_layer3'),
                student_feature=dict(from_student=True, recorder='decoder_block4'),
                teacher_feature=dict(from_student=False, recorder='up_layer3_in', data_idx=0),
            ),
        )))

find_unused_parameters = True

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='mskdcakd_unet_base_unet_small-1000e'),
        )
]
visualizer.update(
    dict(type=SegLocalVisualizer,
         vis_backends=vis_backends,
         name='visualizer'))
