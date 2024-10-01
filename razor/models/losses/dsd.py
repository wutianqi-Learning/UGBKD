import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from monai.networks import one_hot
from monai.networks.blocks.convolutions import Convolution
from monai.metrics.utils import get_mask_edges
from .boundary_loss import BoundaryKDV1, BoundaryKDV4, \
    BoundaryKDV5, BoundaryKDV7, BoundaryKDV9
from .hd_loss import LogHausdorffDTLoss
from .hd_loss_v2 import LogHausdorffDTLossV2


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SepConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 affine=True,
                 instance=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class SepTransConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 affine=True,
                 instance=True):
        #   depthwise and pointwise trans-convolution, upsample by 2
        super(SepTransConv, self).__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose3d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=padding, groups=channel_in, bias=False),
            nn.Conv3d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_in, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_in, affine=affine),
            nn.PReLU(),
            nn.Conv3d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv3d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(channel_out, affine=affine) if instance is True else nn.BatchNorm3d(
                channel_out, affine=affine),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.op(x)


class TransConv(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super(TransConv, self).__init__()
        self.op = nn.Sequential(
            nn.ConvTranspose3d(channel_in,
                               channel_out,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=padding),
            nn.InstanceNorm3d(channel_out),
            nn.PReLU()
        )

    def forward(self, x):
        return self.op(x)


class DSDLoss(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 reduction: str = 'batchmean',
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    out_chans = self.num_classes
                else:
                    out_chans = in_chans // 2
                up_sample_blks.append(SepTransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            nn.Conv3d(self.num_classes, num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        # target_mask = one_hot(label, num_classes=self.num_classes)

        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        gt_labels = label.contiguous().view(-1).long()
        gt_mask = self._get_gt_mask(logits_student, gt_labels)

        tckd_loss = self._get_tckd_loss(logits_student, logits_teacher, gt_labels, gt_mask)
        nckd_loss = self._get_nckd_loss(logits_student, logits_teacher, gt_mask)

        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        return self.loss_weight * loss

    def _get_nckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _get_tckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_labels: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        non_gt_mask = self._get_non_gt_mask(preds_S, gt_labels)
        s_tckd = F.softmax(preds_S / self.tau, dim=1)
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, gt_mask, non_gt_mask))
        mask_teacher = self._cat_mask(t_tckd, gt_mask, non_gt_mask)
        return self._kl_loss(mask_student, mask_teacher)

    def _kl_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau ** 2
        return kl_loss

    def _cat_mask(
            self,
            tckd: torch.Tensor,
            gt_mask: torch.Tensor,
            non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()


class DSDLoss2(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 reduction: str = 'mean',
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss2, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    out_chans = self.num_classes
                else:
                    out_chans = in_chans // 2
                up_sample_blks.append(SepTransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            nn.Conv3d(self.num_classes, num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher):
        logits_student = self.projector(feat_student)

        softmax_pred_T = F.softmax(logits_teacher / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1,
                                                                                                              self.num_classes)
        logsoftmax_preds_S = F.log_softmax(logits_student / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(
            -1, self.num_classes)
        loss = (self.tau ** 2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='mean')

        return self.loss_weight * loss


class DSDLoss3(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 reduction: str = 'mean',
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss3, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    out_chans = self.num_classes
                else:
                    out_chans = in_chans // 2
                up_sample_blks.append(SepTransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            nn.Conv3d(self.num_classes, num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher):
        logits_student = self.projector(feat_student)

        softmax_pred_T = F.softmax(logits_teacher / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(-1,
                                                                                                              self.num_classes)
        logsoftmax_preds_S = F.log_softmax(logits_student / self.tau, dim=1).permute(0, 2, 3, 4, 1).contiguous().view(
            -1, self.num_classes)
        loss = (self.tau ** 2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction='mean')

        return self.loss_weight * loss


class DSDLoss4(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 reduction: str = 'batchmean',
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss4, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
        )
        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        # target_mask = one_hot(label, num_classes=self.num_classes)

        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        gt_labels = label.contiguous().view(-1).long()
        gt_mask = self._get_gt_mask(logits_student, gt_labels)

        tckd_loss = self._get_tckd_loss(logits_student, logits_teacher, gt_labels, gt_mask)
        nckd_loss = self._get_nckd_loss(logits_student, logits_teacher, gt_mask)

        loss = self.alpha * tckd_loss + self.beta * nckd_loss
        return self.loss_weight * loss

    def _get_nckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _get_tckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_labels: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        non_gt_mask = self._get_non_gt_mask(preds_S, gt_labels)
        s_tckd = F.softmax(preds_S / self.tau, dim=1)
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, gt_mask, non_gt_mask))
        mask_teacher = self._cat_mask(t_tckd, gt_mask, non_gt_mask)
        return self._kl_loss(mask_student, mask_teacher)

    def _kl_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau ** 2
        return kl_loss

    def _cat_mask(
            self,
            tckd: torch.Tensor,
            gt_mask: torch.Tensor,
            non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()


class DSDLoss5(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 tau: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(DSDLoss5, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        N, C, H, W, D = logits_student.shape
        target_mask = one_hot(label, num_classes=self.num_classes).view(-1, H * W * D)

        pred_student = F.softmax(logits_student.view(-1, H * W * D) / self.tau, dim=1)
        pred_teacher = F.softmax(logits_teacher.view(-1, H * W * D) / self.tau, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student))
        loss = self.loss_weight * loss / (C * N)

        return loss


class DSDLoss6(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 reduction: str = 'batchmean',
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss6, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(SepTransConv(in_chans, out_chans))
                in_chans //= 2
        else:
            up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector = nn.Sequential(
            *up_sample_blks,
            # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
        )

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        # target_mask = one_hot(label, num_classes=self.num_classes)

        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        gt_labels = label.contiguous().view(-1).long()
        gt_mask = self._get_gt_mask(logits_student, gt_labels)

        tckd_loss = self._get_tckd_loss(logits_student, logits_teacher, gt_labels, gt_mask)
        nckd_loss = self._get_nckd_loss(logits_student, logits_teacher, gt_mask)

        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        return self.loss_weight * loss

    def _get_nckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _get_tckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_labels: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        non_gt_mask = self._get_non_gt_mask(preds_S, gt_labels)
        s_tckd = F.softmax(preds_S / self.tau, dim=1)
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_tckd, gt_mask, non_gt_mask))
        mask_teacher = self._cat_mask(t_tckd, gt_mask, non_gt_mask)
        return self._kl_loss(mask_student, mask_teacher)

    def _kl_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau ** 2
        return kl_loss

    def _cat_mask(
            self,
            tckd: torch.Tensor,
            gt_mask: torch.Tensor,
            non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()


class DSDLoss8(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 kernel_size=3,
                 interpolate=False,
                 bd_include_background=True,
                 hd_include_background=False,
                 one_hot_target=True,
                 sigmoid=False,
                 softmax=True,
                 tau=1,
                 loss_weight: float = 1.0,
                 overall_loss_weight: float = 1.0):
        super(DSDLoss8, self).__init__()
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.tau = tau
        self.bd_include_background = bd_include_background
        self.hd_include_background = hd_include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.overall_loss_weight = overall_loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2

            self.projector = nn.Sequential(
                *up_sample_blks,
                # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
            )
        else:
            if self.num_classes == in_chans:
                self.projector = nn.Identity()
            else:
                self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)
            # up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector.apply(init_weights)

        self.bkd = BoundaryKDV1(
            kernel_size=self.kernel_size,
            tau=tau,
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background)
        self.hd = LogHausdorffDTLoss(
            include_background=hd_include_background,
            to_onehot_y=one_hot_target,
            sigmoid=sigmoid,
            softmax=softmax)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        # loss = self.loss_weight * bkd_loss + (1 - self.loss_weight) * hd_loss
        bkd_loss = bkd_loss * self.overall_loss_weight * self.loss_weight
        hd_loss = hd_loss * self.overall_loss_weight * (1 - self.loss_weight)
        return dict(bkd_loss=bkd_loss, hd_loss=hd_loss)
        # return self.overall_loss_weight * loss

class DKDLoss8(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 kernel_size=3,
                 interpolate=False,
                 bd_include_background=True,
                 hd_include_background=False,
                 one_hot_target=True,
                 sigmoid=False,
                 softmax=True,
                 tau=1,
                 loss_weight: float = 1.0,
                 overall_loss_weight: float = 1.0):
        super(DKDLoss8, self).__init__()
        self.kernel_size = kernel_size
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.tau = tau
        self.bd_include_background = bd_include_background
        self.hd_include_background = hd_include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.overall_loss_weight = overall_loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2

            self.projector = nn.Sequential(
                *up_sample_blks,
                # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
            )
        else:
            if self.num_classes == in_chans:
                self.projector = nn.Identity()
            else:
                self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)
            # up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector.apply(init_weights)

        self.bkd = BoundaryKDV1(
            kernel_size=self.kernel_size,
            tau=tau,
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background)


    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
       
        # loss = self.loss_weight * bkd_loss + (1 - self.loss_weight) * hd_loss
        # bkd_loss = bkd_loss * self.overall_loss_weight * self.loss_weight
        # hd_loss = hd_loss * self.overall_loss_weight * (1 - self.loss_weight)
        return dict(bkd_loss=self.overall_loss_weight * bkd_loss)

class DSDLoss8_BCD(DSDLoss8):

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
        loss = self.loss_weight * bkd_loss
        return self.overall_loss_weight * loss


class DSDLoss9(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 interpolate=False,
                 bd_include_background=True,
                 hd_include_background=False,
                 one_hot_target=True,
                 sigmoid=False,
                 softmax=True,
                 tau=1,
                 loss_weight: float = 1.0,
                 overall_loss_weight: float = 1.0):
        super(DSDLoss9, self).__init__()
        self.interpolate = interpolate
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.tau = tau
        self.bd_include_background = bd_include_background
        self.hd_include_background = hd_include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.overall_loss_weight = overall_loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2

            self.projector = nn.Sequential(
                *up_sample_blks,
                # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
            )
        else:
            if self.num_classes == in_chans:
                self.projector = nn.Identity()
            else:
                self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)
            # up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector.apply(init_weights)

        self.bkd = BoundaryKDV4(
            tau=tau,
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background)
        self.hd = LogHausdorffDTLoss(
            include_background=hd_include_background,
            to_onehot_y=one_hot_target,
            sigmoid=sigmoid,
            softmax=softmax)

    def forward(self, feat_student, logits_teacher, outputs_T, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, outputs_T=outputs_T)
        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        loss = self.loss_weight * bkd_loss + (1 - self.loss_weight) * hd_loss
        return self.overall_loss_weight * loss


class DSDLoss10(DSDLoss9):
    def __init__(self, **kwargs):
        super(DSDLoss10, self).__init__(**kwargs)
        self.bkd = BoundaryKDV5(
            tau=self.tau,
            num_classes=self.num_classes,
            one_hot_target=self.one_hot_target,
            include_background=self.bd_include_background)


class DSDLoss11(DSDLoss9):
    def __init__(self, **kwargs):
        super(DSDLoss11, self).__init__(**kwargs)
        self.bkd = LogHausdorffDTLossV2(
            include_background=self.hd_include_background,
            to_onehot_y=self.one_hot_target,
            sigmoid=self.sigmoid,
            softmax=self.softmax)

        self.hd = LogHausdorffDTLossV2(
            include_background=self.hd_include_background,
            to_onehot_y=self.one_hot_target,
            sigmoid=self.sigmoid,
            softmax=self.softmax)

    def predict(self, logits):
        pred = torch.softmax(logits, 1)
        pred = torch.argmax(pred, 1)
        return pred

    def forward(self, feat_student, logits_teacher, outputs_T, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        pred_student = self.predict(logits_student)
        pred_teacher = self.predict(logits_teacher)

        bkd_loss = torch.tensor(0.).cuda()

        for i in range(1, self.num_classes):
            edges_student, edges_teacher = get_mask_edges(
                pred_student, pred_teacher, label_idx=i, crop=False, always_return_as_numpy=False)
            edges_logits_student = edges_student * logits_student
            edges_logits_teacher = edges_teacher * logits_teacher
            bkd_loss += self.bkd(preds_S=edges_logits_student, preds_T=edges_logits_teacher)
        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher)
        loss = self.loss_weight * bkd_loss + (1 - self.loss_weight) * hd_loss
        return self.overall_loss_weight * loss


class DSDLoss12(DSDLoss8):
    def __init__(self, **kwargs):
        super(DSDLoss12, self).__init__(**kwargs)
        self.bkd = BoundaryKDV7(
            tau=self.tau,
            num_classes=self.num_classes,
            one_hot_target=self.one_hot_target,
            include_background=self.bd_include_background)


class DSDLoss13(DSDLoss8):
    def __init__(self, **kwargs):
        super(DSDLoss13, self).__init__(**kwargs)
        self.bkd = BoundaryKDV9(
            tau=self.tau,
            num_classes=self.num_classes,
            one_hot_target=self.one_hot_target,
            include_background=self.bd_include_background)


class DSDLoss13_HD(DSDLoss13):
    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        hd_loss = hd_loss * self.overall_loss_weight * (1 - self.loss_weight)
        return hd_loss


class DSDLoss13_BKD(DSDLoss13):
    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
        bkd_loss = bkd_loss * self.overall_loss_weight * self.loss_weight
        return bkd_loss
        

class DSDLoss14(DSDLoss8):
    def __init__(self, **kwargs):
        super(DSDLoss14, self).__init__(**kwargs)
        self.bkd = BoundaryKDV7(
            tau=self.tau,
            num_classes=self.num_classes,
            one_hot_target=self.one_hot_target,
            include_background=self.bd_include_background)


class DSDLoss8_HD(DSDLoss8):

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        if self.interpolate:
            logits_student = F.interpolate(logits_student, scale_factor=2, mode='trilinear', align_corners=True)

        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        loss = (1 - self.loss_weight) * hd_loss
        return self.overall_loss_weight * loss


class DSDLoss8_label_supervision(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 num_stages: int,
                 cur_stage: int,
                 bd_include_background=True,
                 hd_include_background=False,
                 one_hot_target=True,
                 sigmoid=False,
                 softmax=True,
                 loss_weight: float = 1.0,
                 overall_loss_weight: float = 1.0):
        super(DSDLoss8_label_supervision, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.overall_loss_weight = overall_loss_weight

        if cur_stage != num_stages:
            up_sample_blk_num = num_stages - cur_stage
            up_sample_blks = []
            for i in range(up_sample_blk_num):
                if i == up_sample_blk_num - 1:
                    # out_chans = self.num_classes
                    up_sample_blks.append(
                        nn.ConvTranspose3d(
                            in_chans,
                            self.num_classes,
                            kernel_size=(3, 3, 3),
                            stride=(2, 2, 2),
                            padding=(1, 1, 1),
                            output_padding=(1, 1, 1)))
                else:
                    out_chans = in_chans // 2
                    up_sample_blks.append(TransConv(in_chans, out_chans))
                in_chans //= 2

            self.projector = nn.Sequential(
                *up_sample_blks,
                # nn.Conv3d(self.num_classes, self.num_classes, 1, 1, 0),
            )
        else:
            self.projector = nn.Identity()
            # up_sample_blks = [nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)]

        self.projector.apply(init_weights)

        self.bkd = BoundaryKDV1(
            num_classes=num_classes,
            one_hot_target=one_hot_target,
            include_background=bd_include_background)
        self.hd = LogHausdorffDTLoss(
            include_background=hd_include_background,
            to_onehot_y=one_hot_target,
            sigmoid=sigmoid,
            softmax=softmax)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        bkd_loss = self.bkd(preds_S=logits_student, preds_T=logits_teacher, gt_labels=label)
        hd_loss = self.hd(preds_S=logits_student, preds_T=logits_teacher, target=label)
        loss = self.loss_weight * bkd_loss + (1 - self.loss_weight) * hd_loss
        return self.overall_loss_weight * loss


class OFALoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 temperature: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(OFALoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, logits_student, logits_teacher, label):
        target_mask = one_hot(label, num_classes=self.num_classes)
        target_mask = target_mask.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)

        return self.loss_weight * loss


class LogOFALoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 temperature: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(LogOFALoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, logits_student, logits_teacher, label):
        target_mask = one_hot(label, num_classes=self.num_classes)
        target_mask = target_mask.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)

        return self.loss_weight * torch.log(loss + 1)


# OFA by CWD
class OFALoss2(nn.Module):
    def __init__(self,
                 num_classes: int,
                 temperature: float = 1.0,
                 eps: float = 1.5,
                 loss_weight: float = 1.0):
        super(OFALoss2, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, logits_student, logits_teacher, label):
        N, C, H, W, D = logits_student.shape
        target_mask = one_hot(label, num_classes=self.num_classes).view(-1, H * W * D)
        # logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        # logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)

        pred_student = F.softmax(logits_student.view(-1, H * W * D) / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher.view(-1, H * W * D) / self.temperature, dim=1)

        prod = (pred_teacher + target_mask) ** self.eps

        loss = torch.sum(- (prod - target_mask) * torch.log(pred_student))
        loss = self.loss_weight * loss / (C * N)

        return loss


class DSDLoss7(nn.Module):
    def __init__(self,
                 in_chans: int,
                 num_classes: int,
                 reduction: str = 'batchmean',
                 tau: float = 1.0,
                 loss_weight: float = 1.0):
        super(DSDLoss7, self).__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.projector = nn.Conv3d(in_chans, self.num_classes, 1, 1, 0)

        self.projector.apply(init_weights)

    def forward(self, feat_student, logits_teacher, label):
        logits_student = self.projector(feat_student)

        # B, C, H, D, W = logits_teacher.shape
        # logits_student = F.interpolate(logits_student, (H, D, W), mode='trilinear')
        B, C, H, D, W = logits_student.shape
        logits_teacher = F.interpolate(logits_teacher, (H, D, W), mode='trilinear')
        label = F.interpolate(label, (H, D, W), mode='trilinear')
        # target_mask = one_hot(label, num_classes=self.num_classes)

        logits_student = logits_student.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        logits_teacher = logits_teacher.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.num_classes)
        gt_labels = label.contiguous().view(-1).long()
        gt_mask = self._get_gt_mask(logits_student, gt_labels)

        nckd_loss = self._get_nckd_loss(logits_student, logits_teacher, gt_mask)

        return self.loss_weight * nckd_loss

    def _get_nckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        s_nckd = F.log_softmax(preds_S / self.tau - 1000.0 * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - 1000.0 * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _kl_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau ** 2
        return kl_loss

    def _cat_mask(
            self,
            tckd: torch.Tensor,
            gt_mask: torch.Tensor,
            non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

    def _get_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    def _get_non_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        target = target.reshape(-1)
        return torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()
