import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import numpy as np

import mmcv
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

from seg.datasets.monai_dataset import SYNAPSE_METAINFO

from monai import transforms
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils.path import mkdir_or_exist

from utils.utils import resample_3d, get_timestamp
from monai.transforms.utils import distance_transform_edt
from monai.metrics.utils import get_mask_edges
from monai.transforms.utils import distance_transform_edt
import cv2


def distance_field(img: torch.Tensor) -> torch.Tensor:
    """Generate distance transform.

    Args:
        img (np.ndarray): input mask as NCHWD or NCHW.

    Returns:
        np.ndarray: Distance field.
    """
    field = torch.zeros_like(img)

    for batch_idx in range(len(img)):
        fg_mask = img[batch_idx] > 0.5

        # For cases where the mask is entirely background or entirely foreground
        # the distance transform is not well defined for all 1s,
        # which always would happen on either foreground or background, so skip
        if fg_mask.any() and not fg_mask.all():
            fg_dist: torch.Tensor = distance_transform_edt(fg_mask)  # type: ignore
            bg_mask = ~fg_mask
            bg_dist: torch.Tensor = distance_transform_edt(bg_mask)  # type: ignore

            field[batch_idx] = fg_dist + bg_dist

    return field


def boundary_pattern_v2(kernel_size=3):
    matrix = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device='cpu')
    matrix = matrix.view(1, 1, kernel_size, kernel_size, kernel_size).cpu()
    return matrix


def get_boundary(gt: torch.Tensor, cls: int) -> torch.Tensor:
    kernel = boundary_pattern_v2(3)
    gt_cls = gt == cls
    boundary = F.conv3d(gt_cls.float().unsqueeze(0), kernel, padding=1)
    boundary[boundary == kernel.sum()] = 0
    boundary[boundary > 0] = 1
    return boundary


if __name__ == '__main__':
    # img_path = 'data/synapse_raw/imagesTr/img0038.nii.gz'
    # label_path = img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')
    # label_path = img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')
    img_path = 'data/WORD/imagesVal/word_0001.nii.gz'
    label_path = img_path.replace('imagesVal', 'labelsVal').replace('img', 'label')
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    label = torch.Tensor(label).cpu()
    cls_idx = 1
    slice_idx = 145


    # slice = label[..., slice_idx]
    # slice_cls = (slice == cls_idx) * 1
    # plt.imshow(slice_cls)
    # plt.show()

    # boundary = get_boundary(label, cls_idx)
    # # plt.imshow(boundary[0][..., slice_idx])
    # # mask_edges = get_mask_edges(label, label, label_idx=cls_idx, crop=False, always_return_as_numpy=False)
    # # plt.imshow(mask_edges[0][..., slice_idx])
    # # plt.show()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_dir = f"save_dirs/edge_word_cls{cls_idx}_slice{slice_idx}"
    label_cls = (torch.Tensor(label).cpu() == cls_idx).float()
    mmcv.imwrite(label_cls[..., slice_idx].numpy() * 255, file_path=f'{save_dir}/{timestamp}_gt.png')
    # mkdir_or_exist(save_dir)
    # mmcv.imwrite(boundary[0][..., slice_idx].numpy()*255, file_path=f'{save_dir}/{timestamp}.png')
    # print(1)

    teacher_label_path = 'save_dirs/swinunetr_base_1000e_word/20240603_165949/predictions/word_0001.nii.gz'
    teacher_label = nib.load(teacher_label_path).get_fdata()
    teacher_label_cls = (torch.Tensor(teacher_label).cpu() == cls_idx).float()
    mmcv.imwrite(teacher_label_cls[..., slice_idx].numpy() * 255, file_path=f'{save_dir}/{timestamp}_teacher.png')
    teacher_dt = distance_field(teacher_label_cls[..., slice_idx].unsqueeze(0).unsqueeze(0))
    teacher_dt_img = teacher_dt[0, 0].numpy()
    # plt.imshow(teacher_dt_img, cmap='plasma')
    # plt.savefig(f'{save_dir}/{timestamp}_teacher_dt.jpg')
    teacher_dt = (teacher_dt - teacher_dt.min()) / (teacher_dt.max() - teacher_dt.min()) * 255
    mmcv.imwrite(teacher_dt[0, 0].numpy(), file_path=f'{save_dir}/{timestamp}_teacher_dt.jpg')

    student_label_path = 'save_dirs/unetmod_tiny_d8_300e_sgd_word_96x96x96/20240603_181227/predictions/word_0001.nii.gz'
    student_label = nib.load(student_label_path).get_fdata()
    student_label_cls = (torch.Tensor(student_label).cpu() == cls_idx).float()
    mmcv.imwrite(student_label_cls[..., slice_idx].numpy() * 255, file_path=f'{save_dir}/{timestamp}_student.png')
    student_dt = distance_field(student_label_cls[..., slice_idx].unsqueeze(0).unsqueeze(0))
    # student_dt_img = student_dt[0, 0].numpy()
    # plt.imshow(student_dt_img, cmap='plasma')
    # plt.savefig(f'{save_dir}/{timestamp}_student_dt.jpg')

    xor = student_label_cls.bool() ^ teacher_label_cls.bool()
    mmcv.imwrite(xor[..., slice_idx].numpy()*255, file_path=f'{save_dir}/{timestamp}_xor.jpg')

    student_dt = (student_dt - student_dt.min()) / (student_dt.max() - student_dt.min()) * 255
    mmcv.imwrite(student_dt[0, 0].numpy(), file_path=f'{save_dir}/{timestamp}_student_dt.jpg')
