# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.runner import load_checkpoint
from mmrazor.models.losses import ChannelWiseDivergence as _CWD
from razor.models.losses import ramps

class ChannelWiseDivergence(_CWD):

    def __init__(self, sigmoid: bool = False, **kwargs):
        super(ChannelWiseDivergence, self).__init__(**kwargs)
        self.sigmoid = sigmoid

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W, D).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W, D).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-3:] == preds_T.shape[-3:]
        N, C, H, W, D = preds_S.shape

        if self.sigmoid:
            sigmoid_pred_T = torch.sigmoid(preds_T.view(-1, H * W * D))

            loss = torch.sum(sigmoid_pred_T *
                             torch.sigmoid(preds_T.view(-1, H * W * D)) -
                             sigmoid_pred_T *
                             torch.sigmoid(preds_S.view(-1, H * W * D)))

            loss = self.loss_weight * loss / (C * N)
        else:
            softmax_pred_T = F.softmax(preds_T.view(-1, H * W * D) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, H * W * D) / self.tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, H * W * D) / self.tau)) * (
                           self.tau ** 2)

            loss = self.loss_weight * loss / (C * N)

        return loss


from razor.models.algorithms.distill.configurable.single_teacher_distill import SingleTeacherDistill_ADE
class ChannelWiseDivergenceWithU(_CWD):

    def __init__(self,
                 dae_network,
                 dae_ckpt,
                 gamma=1.0,
                 epoch=0,
                 total_epochs=1000,
                 hd_epochs=1000,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 hd_include_background=False,
                 one_hot_target=True,
                 softmax=True,
                 sigmoid: bool = False,
                 **kwargs):
        super(ChannelWiseDivergenceWithU, self).__init__(**kwargs)
        self.dae_network = dae_network
        self.dae_ckpt = dae_ckpt
        self.dae_model = self.load_DAE_model(self.dae_ckpt)
        self.gamma = gamma
        self.epoch = epoch
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.sigmoid = sigmoid
        self.criterion_kd = torch.nn.KLDivLoss()
        # self.hd_epochs = hd_epochs
        # self.hd = LogHausdorffDTLoss(
        #     include_background=hd_include_background,
        #     to_onehot_y=one_hot_target,
        #     sigmoid=sigmoid,
        #     softmax=softmax)

    # def load_DAE_model(self, dae_ckpt, detach=True):
    #     # save_mode_path = os.path.join(snapshot_path_DAE)
    #     # model = SegResNet(out_channels=out_channels).cuda()
    #     model = MODELS.build(self.dae_network).cuda()
    #     if dae_ckpt:
    #         _ = load_checkpoint(model, dae_ckpt)
    #         # avoid loaded parameters be overwritten
    #         model._is_init = True
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     return model
    def load_DAE_model(self, dae_ckpt, detach=True):
    
        # save_mode_path = os.path.join(snapshot_path_DAE)
        # model = SegResNet(out_channels=out_channels).cuda()
        dae_network = self.dae_network
        ade = SingleTeacherDistill_ADE(
            architecture=dict(cfg_path=dae_network, pretrained=False),
            teacher=dict(cfg_path=dae_network, pretrained=False), 
            teacher_ckpt=dae_ckpt)
        
        return ade.getmodel()


    def get_current_consistency_weight(self, epoch, consistency=0.1, consistency_rampup=40.0):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W, D).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W, D).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-3:] == preds_T.shape[-3:]

        N, C, H, W, D = preds_S.shape

        # get output of teacher model
        dea_input = F.softmax(preds_T, dim=1)
        dea_input_argmaxed = torch.argmax(dea_input, dim=1)
        dea_input_argmaxed = dea_input_argmaxed[:, None, :, :, :].type(torch.FloatTensor).cuda()

        # get output of student model
        dea_stu_input = F.softmax(preds_S, dim=1)
        dea_stu_input_argmaxed = torch.argmax(dea_stu_input, dim=1)
        dea__stu_input_argmaxed = dea_stu_input_argmaxed[:, None, :, :, :].type(torch.FloatTensor).cuda()
        
        self.dae_model.eval()
        dae_outputs = self.dae_model(dea_input_argmaxed)
        dae_stu_outputs = self.dae_model(dea__stu_input_argmaxed)

        # get mask of teacher 
        uncertainty = (F.softmax(preds_T, dim=1) - F.softmax(dae_outputs, dim=1)) ** 2
        consistency_weight = self.get_current_consistency_weight(self.epoch, self.consistency, self.consistency_rampup)
        certainty = torch.exp(-1.0 * self.gamma * uncertainty)
        mask = certainty.float()
        
        # get mask of student 
        uncertainty_stu = (F.softmax(preds_S, dim=1) - F.softmax(dae_stu_outputs, dim=1)) ** 2
        certainty_stu = torch.exp(self.gamma * uncertainty_stu)
        mask_stu = certainty_stu.float()

        if self.sigmoid:
            sigmoid_pred_T = torch.sigmoid(preds_T.view(-1, H * W * D))

            loss = torch.sum(sigmoid_pred_T *
                             torch.sigmoid(preds_T.view(-1, H * W * D)) -
                             sigmoid_pred_T *
                             torch.sigmoid(preds_S.view(-1, H * W * D)))

            loss = self.loss_weight * loss / (C * N)
        else:
            softmax_pred_T = F.softmax(preds_T.view(-1, H * W * D) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, H * W * D) / self.tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, H * W * D) / self.tau)) * (self.tau ** 2)

            loss = self.loss_weight * loss / (C * N)

        # uncertainty kl kd
        uncertainty_stu = (preds_S - dae_outputs) ** 2
        uncertainty = (preds_T - dae_outputs) ** 2
        loss_u_kl = self.criterion_kd(
            F.log_softmax(uncertainty_stu, dim=1),
            F.softmax(uncertainty, dim=1))
        
        loss_kl = self.criterion_kd(
            F.log_softmax(preds_S / self.tau, dim=1),
            F.softmax(preds_T / self.tau, dim=1)) * (self.tau ** 2)
        # loss_mse = (uncertainty_stu - uncertainty) ** 2
        consistency_weight_u = self.get_current_consistency_weight(self.epoch,self.consistency, self.consistency_rampup)
        loss = torch.sum(mask * torch.exp(loss_kl) * loss) / (2 * torch.sum(mask) + 1e-16) 
        # loss = self.loss_weight * (consistency_weight * (loss + loss_kl))
        
        return dict(guide_loss=self.loss_weight * loss,
                    noice_loss=consistency_weight_u * torch.log(2 * torch.sum(mask) + 1e-16),
                    u_loss=self.loss_weight * consistency_weight_u * 10 *loss_u_kl)

      
class ChannelWiseDivergenceOnlyGuide(_CWD):

    def __init__(self,
                 dae_network,
                 dae_ckpt,
                 gamma=1.0,
                 epoch=0,
                 total_epochs=1000,
                 hd_epochs=1000,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 hd_include_background=False,
                 one_hot_target=True,
                 softmax=True,
                 sigmoid: bool = False,
                 **kwargs):
        super(ChannelWiseDivergenceOnlyGuide, self).__init__(**kwargs)
        self.dae_network = dae_network
        self.dae_ckpt = dae_ckpt
        self.dae_model = self.load_DAE_model(self.dae_ckpt)
        self.gamma = gamma
        self.epoch = epoch
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.sigmoid = sigmoid
        self.criterion_kd = torch.nn.KLDivLoss()
        # self.hd_epochs = hd_epochs
        # self.hd = LogHausdorffDTLoss(
        #     include_background=hd_include_background,
        #     to_onehot_y=one_hot_target,
        #     sigmoid=sigmoid,
        #     softmax=softmax)

    def load_DAE_model(self, dae_ckpt, detach=True):
        # save_mode_path = os.path.join(snapshot_path_DAE)
        # model = SegResNet(out_channels=out_channels).cuda()
        model = MODELS.build(self.dae_network).cuda()
        if dae_ckpt:
            _ = load_checkpoint(model, dae_ckpt)
            # avoid loaded parameters be overwritten
            model._is_init = True
        for param in model.parameters():
            param.requires_grad = False
        return model

    def get_current_consistency_weight(self, epoch, consistency=0.1, consistency_rampup=40.0):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242

        return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W, D).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W, D).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-3:] == preds_T.shape[-3:]

        N, C, H, W, D = preds_S.shape

        # get output of teacher model
        dea_input = F.softmax(preds_T, dim=1)
        dea_input_argmaxed = torch.argmax(dea_input, dim=1)
        dea_input_argmaxed = dea_input_argmaxed[:, None, :, :, :].type(torch.FloatTensor).cuda()

        # get output of student model
        dea_stu_input = F.softmax(preds_S, dim=1)
        dea_stu_input_argmaxed = torch.argmax(dea_stu_input, dim=1)
        dea__stu_input_argmaxed = dea_stu_input_argmaxed[:, None, :, :, :].type(torch.FloatTensor).cuda()
        
        self.dae_model.eval()
        dae_outputs = self.dae_model(dea_input_argmaxed)
        dae_stu_outputs = self.dae_model(dea__stu_input_argmaxed)

        # get mask of teacher 
        uncertainty = (F.softmax(preds_T, dim=1) - F.softmax(dae_outputs, dim=1)) ** 2
        consistency_weight = self.get_current_consistency_weight(self.epoch, self.consistency, self.consistency_rampup)
        certainty = torch.exp(-1.0 * self.gamma * uncertainty)
        mask = certainty.float()
        
        # get mask of student 
        uncertainty_stu = (F.softmax(preds_S, dim=1) - F.softmax(dae_stu_outputs, dim=1)) ** 2
        certainty_stu = torch.exp(self.gamma * uncertainty_stu)
        mask_stu = certainty_stu.float()

        if self.sigmoid:
            sigmoid_pred_T = torch.sigmoid(preds_T.view(-1, H * W * D))

            loss = torch.sum(sigmoid_pred_T *
                             torch.sigmoid(preds_T.view(-1, H * W * D)) -
                             sigmoid_pred_T *
                             torch.sigmoid(preds_S.view(-1, H * W * D)))

            loss = self.loss_weight * loss / (C * N)
        else:
            softmax_pred_T = F.softmax(preds_T.view(-1, H * W * D) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            loss = torch.sum(softmax_pred_T *
                             logsoftmax(preds_T.view(-1, H * W * D) / self.tau) -
                             softmax_pred_T *
                             logsoftmax(preds_S.view(-1, H * W * D) / self.tau)) * (self.tau ** 2)

            loss = self.loss_weight * loss / (C * N)

       
        loss = torch.sum(mask * loss) / (2 * torch.sum(mask) + 1e-16) 
        # loss = self.loss_weight * (consistency_weight * (loss + loss_kl))
        
        return dict(guide_loss=self.loss_weight * loss)
   