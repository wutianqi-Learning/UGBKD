# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses.huasdorff_distance_loss import HuasdorffDisstanceLoss
from cucim.skimage.morphology import binary_erosion
from monai.utils import convert_to_cupy, convert_to_tensor
from monai.metrics.utils import get_mask_edges
from .hd_loss_v2 import LogHausdorffDTLossV2
from .bpkd import get_mask_edges as get_mask_edges_v2
# from mmseg.models.decode_heads import PIDHead
# from mmseg.models.backbones import PIDNet
# from mmseg.datasets.transforms import GenerateEdge
def L2(f_):
    return (((f_**2).sum())**0.5) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    return torch.einsum('m,n->mn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


def boundary_pattern(kernel_size=3):
    matrix = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device='cpu')
    matrix[1:-1, 1:-1, 1:-1] = 0
    matrix = matrix.view(1, 1, kernel_size, kernel_size, kernel_size).cuda()
    return matrix


def boundary_pattern_v2(kernel_size=3):
    matrix = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device='cpu')
    matrix = matrix.view(1, 1, kernel_size, kernel_size, kernel_size).cuda()
    return matrix


def get_edges(
    seg_label: torch.Tensor,
    label_idx: int = 1,
) -> torch.Tensor:
    """
    Compute edges from binary segmentation masks. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    Args:
        seg_label: the predicted binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
    """
    converter = partial(convert_to_tensor, device=seg_label.device)

    # If not binary images, convert them
    if seg_label.dtype not in (bool, torch.bool):
        seg_label = seg_label == label_idx

    seg_label = convert_to_cupy(seg_label, dtype=bool)  # type: ignore[arg-type]
    edges_label = binary_erosion(seg_label) ^ seg_label
    return converter(edges_label, dtype=bool)  # type: ignore


class BoundaryKDV1(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 kernel_size: int = 3,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV1, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern_v2(kernel_size)
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        loss = torch.tensor(0.).cuda()
        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(gt_labels[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0)) * (self.temperature**2)
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        return self.loss_weight * loss


class BoundaryKDV3(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV3, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.one_hot_target = one_hot_target
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        # loss = torch.tensor(0.).cuda()
        all_loss = []
        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(gt_labels[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    l = F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0))
                else:
                    l = F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )
                all_loss.append(l.unsqueeze(0))
        if len(all_loss) == 0:
            return torch.tensor(0.).cuda()
        loss = torch.cat(all_loss, dim=0)
        loss = torch.mean(loss)
        return self.loss_weight * loss


class BoundaryKDV4(nn.Module):
    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV4, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern_v2()
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)

        # img_path = gt.meta['filename_or_obj']
        # affine = gt.meta['original_affine']
        #
        # import numpy as np
        # import os
        # import os.path as osp
        # import nibabel as nib
        #
        # outputs = boundary.cpu().numpy().astype(np.uint8)[0]
        # labels = gt.cpu().numpy().astype(np.uint8)[0]
        #
        # save_dirs = os.path.join(
        #     'save_dirs', 'boundary',
        #     f"{0}_{osp.basename(img_path.replace('label', 'boundary'))}")
        # nib.save(
        #     nib.Nifti1Image(outputs.astype(np.uint8), affine),
        #     save_dirs)
        #
        # save_dirs = os.path.join(
        #     'save_dirs', 'boundary',
        #     f'{0}_{osp.basename(img_path)}')
        # nib.save(
        #     nib.Nifti1Image(labels.astype(np.uint8), affine),
        #     save_dirs)

        # _, H, W, D = boundary.shape
        #
        # boundary_ = boundary
        #
        # boundary_[0, 0, ...][boundary_[0, 0, ...] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa
        # boundary_[0, H - 1, ...][boundary_[0, H - 1, ...] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa
        # boundary_[0, 0:H - 1, 0, ...][boundary_[0, 0:H - 1, 0, ...] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa
        # boundary_[0, 0:H - 1, W - 1, ...][boundary_[0, 0:H - 1, W - 1, ...] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa
        # boundary_[0, 0:H - 1, 0:W - 1, 0][boundary_[0, 0:H - 1, 0:W - 1, 0] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa
        # boundary_[0, 0:H - 1, 0:W - 1, D - 1][boundary_[0, 0:H - 1, 0:W - 1, D - 1] == (self.kernel.sum() - self.kernel[0, 0, 0, ...].sum())] == 0 # noqa

        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1

        # outputs = boundary.cpu().numpy().astype(np.uint8)[0]
        #
        # save_dirs = os.path.join(
        #     'save_dirs', 'boundary',
        #     f"{0}_{osp.basename(img_path.replace('label', 'extract_boundary'))}")
        # nib.save(
        #     nib.Nifti1Image(outputs.astype(np.uint8), affine),
        #     save_dirs)

        return boundary

    def forward(self, preds_S, preds_T, outputs_T):
        # outputs_T = torch.softmax(outputs_T, 1)
        targets_T = torch.argmax(outputs_T, 1).detach().unsqueeze(1)

        batch_size, C, H, W, D = preds_S.shape
        loss = torch.tensor(0.).cuda()

        for bs in range(batch_size):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(self.num_classes):
                if cls == 0 and not self.include_background:
                    continue
                boundary = self.get_boundary(targets_T[bs].detach().clone(), cls)
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0)) * (self.temperature**2)
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        # boundary = torch.zeros_like(preds_S)
        #
        # for bs in range(batch_size):
        #     for cls in range(self.num_classes):
        #         boundary_b_c = self.get_boundary(gt_labels[bs].detach().clone(), cls)
        #         boundary[bs, cls, ...] = boundary_b_c.squeeze(0)

        # boundary_labels = torch.argmax(boundary, dim=1)
        #
        # img_path = gt_labels.meta['filename_or_obj']
        # shape = gt_labels.meta['spatial_shape']
        # affine = gt_labels.meta['original_affine']
        #
        # import numpy as np
        # from utils.utils import resample_3d
        # import os
        # import os.path as osp
        # import nibabel as nib
        #
        # for bs in range(batch_size):
        #
        #     outputs = boundary_labels.cpu().numpy().astype(np.uint8)[bs]
        #     labels = gt_labels.cpu().numpy().astype(np.uint8)[bs][0]
        #     # val_outputs = resample_3d(outputs, shape[bs])
        #
        #     save_dirs = os.path.join(
        #         'save_dirs', 'boundary',
        #         f"{bs}_{osp.basename(img_path[bs].replace('label', 'boundary'))}")
        #     nib.save(
        #         nib.Nifti1Image(outputs.astype(np.uint8), affine[bs]),
        #         save_dirs)
        #
        #     save_dirs = os.path.join(
        #         'save_dirs', 'boundary',
        #         f'{bs}_{osp.basename(img_path[bs])}')
        #     nib.save(
        #         nib.Nifti1Image(labels.astype(np.uint8), affine[bs]),
        #         save_dirs)

        return self.loss_weight * loss


class BoundaryKDV5(BoundaryKDV4):
    def __init__(self,
                 **kwargs):
        super(BoundaryKDV5, self).__init__(**kwargs)
        self.include_background = False

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        if self.one_hot_target:
            gt_cls = gt == cls
        else:
            gt_cls = gt[cls, ...].unsqueeze(0)
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)

        # remove patch boundary
        patch_boundary_mask = torch.zeros_like(boundary)
        patch_boundary_mask[:, 1:-1, 1:-1, 1:-1] = 1
        boundary = boundary * patch_boundary_mask

        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1

        return boundary


class BoundaryKDV6(nn.Module):
    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV6, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = LogHausdorffDTLossV2(
            include_background=include_background,
            to_onehot_y=one_hot_target,
            softmax=False)

    def predict(self, logits):
        pred = torch.softmax(logits, 1)
        pred = torch.argmax(pred, 1)
        return pred

    def forward(self, logits_student, logits_teacher):
        preds_student = self.predict(logits_student)
        preds_teacher = self.predict(logits_teacher)

        loss = torch.tensor(0.).cuda()

        for bs in range(logits_student.shape[0]):
            for cls in range(1, self.num_classes):
                logits_student_bs_cls = logits_student[bs, cls]
                # logits_teacher_bs_cls = logits_teacher[bs, cls]
                edges_student, edges_teacher = get_mask_edges(
                    preds_student[bs], preds_teacher[bs], label_idx=cls, crop=False, always_return_as_numpy=False)
                edges_logits_student = edges_student * logits_student_bs_cls
                edges_preds_teacher = edges_teacher * preds_teacher[bs]
                loss += self.criterion_kd(
                    preds_S=edges_logits_student.unsqueeze(0).unsqueeze(0),
                    preds_T=edges_preds_teacher.unsqueeze(0).unsqueeze(0))

                # img_path = preds_teacher.meta['filename_or_obj']
                # affine = preds_teacher.meta['original_affine']
                #
                # import numpy as np
                # import os
                # import os.path as osp
                # import nibabel as nib
                #
                # edges_teacher_ = edges_teacher.cpu().numpy().astype(np.uint8)
                # edges_student_ = edges_student.cpu().numpy().astype(np.uint8)
                # preds_teacher_ = preds_teacher.cpu().numpy().astype(np.uint8)[bs]
                #
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{osp.basename(img_path[bs].replace('img', 'edges_teacher'))}")
                # nib.save(
                #     nib.Nifti1Image(edges_teacher_.astype(np.uint8), affine[bs]),
                #     save_dirs)
                #
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{osp.basename(img_path[bs].replace('img', 'edges_student'))}")
                # nib.save(
                #     nib.Nifti1Image(edges_student_.astype(np.uint8), affine[bs]),
                #     save_dirs)
                #
                # save_dirs = os.path.join(
                #     'save_dirs', 'boundary',
                #     f"{bs}_{osp.basename(img_path[bs].replace('img', 'preds_teacher'))}")
                # nib.save(
                #     nib.Nifti1Image(preds_teacher_.astype(np.uint8), affine[bs]),
                #     save_dirs)

        return (1 - self.loss_weight) * loss


class BoundaryKDV7(nn.Module):
    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 one_hot_target: bool = True,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BoundaryKDV7, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.one_hot_target = one_hot_target
        self.include_background = include_background
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_S, preds_T, gt_labels):
        loss = torch.tensor(0.).cuda()

        for bs in range(preds_S.shape[0]):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(1, self.num_classes):
                boundary = get_edges(gt_labels[bs, 0].detach().clone(), label_idx=cls).long()
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0)) * (self.temperature**2)
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        return self.loss_weight * loss


class BodyKDV8(nn.Module):
    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 include_background: bool = True,
                 loss_weight: float = 1.0):
        super(BodyKDV8, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.include_background = include_background

        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds_S, preds_T, gt_labels):
        loss = torch.tensor(0.).cuda()

        for bs in range(preds_S.shape[0]):
            gt_labels_bs  = gt_labels[bs, 0].detach().clone()
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(1, self.num_classes):
                body_mask = (gt_labels_bs == cls).long().view(-1)
                body_idxs = (body_mask == 1).nonzero()

                if body_idxs.sum() == 0:
                    continue

                body_S = preds_S_i[:, body_idxs].squeeze(-1)
                body_T = preds_T_i[:, body_idxs].squeeze(-1)

                loss += F.kl_div(
                    F.log_softmax(body_S / self.temperature, dim=0),
                    F.softmax(body_T / self.temperature, dim=0)) * (self.temperature**2)

        return self.loss_weight * loss


class BoundaryKDV9(BoundaryKDV7):
    def forward(self, preds_S, preds_T, gt_labels):
        loss = torch.tensor(0.).cuda()

        for bs in range(preds_S.shape[0]):
            preds_S_i = preds_S[bs].contiguous().view(preds_S.shape[1], -1)
            preds_T_i = preds_T[bs].contiguous().view(preds_T.shape[1], -1)
            preds_T_i.detach()
            for cls in range(1, self.num_classes):
                boundary = get_mask_edges_v2(gt_labels[bs, 0].detach().clone(), label_idx=cls).long()
                boundary = boundary.view(-1)
                idxs = (boundary == 1).nonzero()
                if idxs.sum() == 0:
                    continue
                boundary_S = preds_S_i[:, idxs].squeeze(-1)
                boundary_T = preds_T_i[:, idxs].squeeze(-1)
                if self.one_hot_target:
                    loss += F.kl_div(
                        F.log_softmax(boundary_S / self.temperature, dim=0),
                        F.softmax(boundary_T / self.temperature, dim=0)) * (self.temperature**2)
                else:
                    loss += F.mse_loss(
                        torch.sigmoid(boundary_S),
                        torch.sigmoid(boundary_T)
                    )

        return self.loss_weight * loss


class BoundaryKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 loss_weight: float = 1.0):
        super(BoundaryKD, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.criterion_kd = torch.nn.KLDivLoss()

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W, D = fea_0.size()

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)

        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def get_boundary(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        gt_cls = gt == cls
        boundary = F.conv3d(gt_cls.float(), self.kernel, padding=1)
        boundary[boundary == self.kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary

    def sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits / self.temperature, dim=1)
        p_t = F.softmax(t_logits / self.temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.temperature ** 2
        return sim_dis

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        preds_S = preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)

        loss = torch.tensor(0.).cuda()

        preds_T.detach()
        for cls in range(1, self.num_classes):
            boundary = self.get_boundary(gt_labels.detach().clone(), cls)
            boundary = boundary.view(-1)
            idxs = (boundary == 1).nonzero()
            if idxs.sum() == 0:
                continue
            # num_classes x num_pixels
            boundary_S = preds_S[idxs, :].squeeze(1)
            boundary_T = preds_T[idxs, :].squeeze(1)
            loss += F.kl_div(
                F.log_softmax(boundary_S / self.temperature, dim=1),
                F.softmax(boundary_T / self.temperature, dim=1))
        loss = loss / (self.num_classes - 1)
        return self.loss_weight * loss


class AreaKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self,
                 tau: float = 1.0,
                 num_classes: int = 14,
                 loss_weight: float = 1.0):
        super(AreaKD, self).__init__()
        self.temperature = tau
        self.loss_weight = loss_weight
        self.kernel = boundary_pattern()
        self.num_classes = num_classes
        self.criterion_kd = torch.nn.KLDivLoss()

    def get_area(self, gt: torch.Tensor, cls: int) -> torch.Tensor:
        area = gt == cls
        area = area.float()
        return area

    def forward(self, preds_S, preds_T, gt_labels):
        batch_size, C, H, W, D = preds_S.shape
        # loss = torch.tensor(0.).cuda()
        preds_S = preds_S.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        gt_labels = gt_labels.view(-1, 1)
        onehot_mask = torch.zeros_like(preds_S).scatter_(1, gt_labels.long(), 1).bool()
        # preds_T.detach()

        loss = F.kl_div(
            F.log_softmax(preds_S * onehot_mask / self.temperature, dim=1),
            F.softmax(preds_T * onehot_mask / self.temperature, dim=1),
            size_average=False,
            reduction='batchmean') * self.temperature**2

        # for cls in range(1, self.num_classes):
        #     area = self.get_area(gt_labels.detach().clone(), cls)
        #     area = area.view(-1)
        #     idxs = (area == 1).nonzero()
        #     if idxs.sum() == 0:
        #         continue
        #     area_S = preds_S[idxs, cls].squeeze(-1)
        #     area_T = preds_T[idxs, cls].squeeze(-1)
        #     loss += F.kl_div(
        #         F.log_softmax(area_S / self.temperature, dim=0),
        #         F.softmax(area_T / self.temperature, dim=0))

        return self.loss_weight * loss

