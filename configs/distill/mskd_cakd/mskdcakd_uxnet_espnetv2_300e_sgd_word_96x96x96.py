from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder
from razor.models.losses.mskd_cakd import CAKD, KD
from razor.models.architectures.connectors.conv3d_connector import Conv3DConnector

with read_base():
    from ..._base_.datasets.word import *  # noqa
    from ..._base_.schedules.schedule_300e_sgd import *  # noqa
    from ..._base_.monai_runtime import *  # noqa
    # import teacher model and student model
    from ...uxnet.uxnet_b1_2000e_adamw_noscheduler_word_96x96x96 import model as teacher_model  # noqa
    from ...espnetv2.espnetv2_300e_sgd_word_96x96x96 import model as student_model  # noqa

teacher_ckpt = 'ckpts/uxnet_b1_2000e_adamw_noscheduler_word_96x96x96/best_Dice_85-29_epoch_1900.pth'  # noqa: E501

num_classes = 17

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_model, pretrained=False),
    teacher=dict(cfg_path=teacher_model, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            # up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l1'),#64x6
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l2'),#48x12
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l3'),#32x24
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.bu_dec_l4'),#14x48
            # up_layer3_in=dict(type=ModuleInputsRecorder, source='segmentor.backbone.up_layer3'),
            up_layer4=dict(type=ModuleOutputsRecorder, source='segmentor')#14x96
        ),
        teacher_recorders=dict(
            up_layer1=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder5'),#384x12
            up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder4'),#192x24
            up_layer3=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder3'),#96x48
            # up_layer2=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder2'),#48x96
            # up_layer3_in=dict(type=ModuleOutputsRecorder, source='segmentor.backbone.decoder1'),#48x96
            up_layer4=dict(type=ModuleOutputsRecorder, source='segmentor')#14x96
        ),
        distill_losses=dict(
            loss_cakd1=dict(type=CAKD, loss_weight=0.2),
            loss_cakd2=dict(type=CAKD, loss_weight=0.2),
            loss_cakd3=dict(type=CAKD, loss_weight=0.2),
            loss_cakd4=dict(type=CAKD, loss_weight=0.4),
            loss_kd1=dict(type=KD, loss_weight=0.4),
            loss_kd2=dict(type=KD, loss_weight=0.4),
            loss_kd3=dict(type=KD, loss_weight=0.4),
            loss_kd4=dict(type=KD, loss_weight=0.8),
        ),
        connectors=dict(
            pred_t_1=dict(
                type=Conv3DConnector,
                args=dict(in_channels=384,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
            pred_t_2=dict(
                type=Conv3DConnector,
                args=dict(in_channels=192,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
            pred_t_3=dict(
                type=Conv3DConnector,
                args=dict(in_channels=96,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
            pred_s_1=dict(
                type=Conv3DConnector,
                args=dict(in_channels=48,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
            pred_s_2=dict(
                type=Conv3DConnector,
                args=dict(in_channels=32,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
            pred_s_3=dict(
                type=Conv3DConnector,
                args=dict(in_channels=num_classes,
                          out_channels=num_classes,
                          kernel_size=1,
                          padding=0)),
        ),
        loss_forward_mappings=dict(
            loss_cakd1=dict(
                student_outputs=dict(from_student=True, recorder='up_layer1', connector='pred_s_1'),
                teacher_outputs=dict(from_student=False, recorder='up_layer1', connector='pred_t_1'),
            ),
            loss_cakd2=dict(
                student_outputs=dict(from_student=True, recorder='up_layer2', connector='pred_s_2'),
                teacher_outputs=dict(from_student=False, recorder='up_layer2', connector='pred_t_2'),
            ),
            loss_cakd3=dict(
                student_outputs=dict(from_student=True, recorder='up_layer3', connector='pred_s_3'),
                teacher_outputs=dict(from_student=False, recorder='up_layer3', connector='pred_t_3'),
            ),
            loss_cakd4=dict(
                student_outputs=dict(from_student=True, recorder='up_layer4'),
                teacher_outputs=dict(from_student=False, recorder='up_layer4'),
            ),
            loss_kd1=dict(
                student_outputs=dict(from_student=True, recorder='up_layer1', connector='pred_s_1'),
                teacher_outputs=dict(from_student=False, recorder='up_layer1', connector='pred_t_1'),
            ),
            loss_kd2=dict(
                student_outputs=dict(from_student=True, recorder='up_layer2', connector='pred_s_2'),
                teacher_outputs=dict(from_student=False, recorder='up_layer2', connector='pred_t_2'),
            ),
            loss_kd3=dict(
                student_outputs=dict(from_student=True, recorder='up_layer3', connector='pred_s_3'),
                teacher_outputs=dict(from_student=False, recorder='up_layer3', connector='pred_t_3'),
            ),
            loss_kd4=dict(
                student_outputs=dict(from_student=True, recorder='up_layer4'),
                teacher_outputs=dict(from_student=False, recorder='up_layer4'),
            ),
        )))

find_unused_parameters = True
