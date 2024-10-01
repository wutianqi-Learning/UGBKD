import os
import os.path as osp

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

    return field, fg_dist, bg_dist

if __name__ == '__main__':
    img_path = 'data/synapse_raw/imagesTr/img0038.nii.gz'
    label_path = img_path.replace('imagesTr', 'labelsTr').replace('img', 'label')

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    cls_idx = 1
    slice_idx = 78
    palette = SYNAPSE_METAINFO['palette']

    teacher_preds = nib.load(
        'save_dirs/swinunetr_base_5000e_synapse/20240425_184059/predictions/pred0038.nii.gz').get_fdata()

    # student_preds = nib.load(
    #     'save_dirs/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96/20240425_184409/predictions/pred0038.nii.gz').get_fdata()
    student_preds = nib.load(
        'save_dirs/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96/20240425_214346/predictions/pred0038.nii.gz').get_fdata()
    mask = np.expand_dims(teacher_preds[..., slice_idx] == cls_idx, -1)
    slice = mask * 255
    mmcv.imwrite(img=slice, file_path=f'save_dirs/paper_vis/preds_teacher_38_{slice_idx}_{cls_idx}.png')
    mask = teacher_preds == cls_idx
    fg_dist = distance_transform_edt(mask)
    fg_dist_slice = fg_dist[..., slice_idx]
    fg_dist_slice = (fg_dist_slice - fg_dist_slice.min()) / fg_dist_slice.max() * 255
    mmcv.imwrite(img=fg_dist_slice,
                 file_path=f'save_dirs/paper_vis/dt_teacher_38_{slice_idx}_{cls_idx}_fg_dist_slice.png')

    mask = np.expand_dims(student_preds[..., slice_idx] == cls_idx, -1)
    slice = mask * 255
    mmcv.imwrite(img=slice, file_path=f'save_dirs/paper_vis/preds_ourkd_student_38_{slice_idx}_{cls_idx}.png')
    mask = student_preds == cls_idx
    fg_dist = distance_transform_edt(mask)
    fg_dist_slice = fg_dist[..., slice_idx]
    fg_dist_slice = (fg_dist_slice - fg_dist_slice.min()) / fg_dist_slice.max() * 255
    mmcv.imwrite(img=fg_dist_slice,
                 file_path=f'save_dirs/paper_vis/dt_ourkd_38_{slice_idx}_{cls_idx}_fg_dist_slice.png')

    transform = transforms.Compose(
        [transforms.LoadImaged(keys=["image"])]
    )
    data = transform({'image': img_path})
    outputs = fg_dist.astype(np.uint8)

    nib.save(
        nib.Nifti1Image(outputs, data['image'].affine),
        'save_dirs/paper_vis/dt_student_38_{slice_idx}_{cls_idx}_3d.nii.gz')
    gt_cls = label[..., slice_idx] == cls_idx
    gt_cls = torch.tensor(gt_cls).unsqueeze(0)
    kernel = torch.ones((3, 3), dtype=torch.float32, device='cpu')
    kernel[1:-1, 1:-1] = 0
    kernel = kernel.view(1, 1, 3, 3)
    boundary = F.conv2d(gt_cls.float(), kernel, padding=1)
    boundary[boundary == kernel.sum()] = 0
    boundary[boundary > 0] = 1
    plt.imshow(boundary[0])
    plt.show()

    boundary_rgb = torch.cat([boundary, boundary, boundary], dim=0)
    boundary_rgb = boundary_rgb.permute(1, 2, 0).numpy()
    boundary_rgb[..., 0] = boundary_rgb[..., 0] * palette[cls_idx][2]  # 213
    boundary_rgb[..., 1] = boundary_rgb[..., 1] * palette[cls_idx][1]   # 239
    boundary_rgb[..., 2] = boundary_rgb[..., 2] * palette[cls_idx][0]  # 255
    mmcv.imwrite(img=boundary_rgb, file_path=f'save_dirs/paper_vis/boundary_{cls_idx}.jpg')

    # teacher_label = nib.load('save_dirs/paper_vis/pred0038_teacher.nii.gz').get_fdata()
    # student_label = nib.load('save_dirs/paper_vis/pred0038_student.nii.gz').get_fdata()
    # plt.imshow(student_label[..., slice_idx])
    # plt.show()
    # plt.imshow(teacher_label[..., slice_idx])
    # plt.show()
    # plt.imshow(label[..., slice_idx])
    # plt.show()
    # print()