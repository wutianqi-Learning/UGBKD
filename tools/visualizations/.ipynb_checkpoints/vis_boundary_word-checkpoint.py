import os
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np

import mmcv
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

from seg.datasets.monai_dataset import WORD_METAINFO

from monai import transforms
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmengine.utils.path import mkdir_or_exist

from utils.utils import resample_3d, get_timestamp
from monai.transforms.utils import distance_transform_edt
from skimage.morphology import binary_erosion
from scipy.ndimage import binary_dilation
from razor.models.losses.boundary_loss import boundary_pattern_v2
from PIL import Image

def get_edges(
    seg_label: np.ndarray,
    label_idx: int = 1,
    use_conv: bool = False,
) -> np.ndarray:

    if seg_label.dtype not in (bool, torch.bool):
        seg_label = seg_label == label_idx
    if use_conv:
        kernel = boundary_pattern_v2(3).cpu()
        seg_label = torch.Tensor(seg_label).unsqueeze(0).unsqueeze(0)
        boundary = F.conv3d(seg_label.float(), kernel, padding=1)
        boundary[boundary == kernel.sum()] = 0
        boundary[boundary > 0] = 1
        return boundary.numpy()[0, 0]

    else:
        edges_label = binary_erosion(seg_label) ^ seg_label
        return edges_label.astype(np.uint8)


def png_to_jpg(img: np.ndarray, palette: list) -> np.ndarray:
    if img.ndim < 3:
        img = np.expand_dims(img, -1)
    assert img.ndim == 3 and len(palette) == 3
    img = np.concatenate([img, img, img], -1)
    img = img * palette
    return img.astype(np.uint8)


def scale(img, a_min=-175, a_max=250, b_min=0.0, b_max=1.0):
    img = (img - a_min) / (a_max - a_min)
    img = img * (b_max - b_min) + b_min
    img = np.clip(img, b_min, b_max)
    return img


def cover(img1, img2, palette, alpha):
    img1[np.where(img2)] = 0
    return (img1 + alpha * png_to_jpg(img2, palette)).astype(np.uint8)


if __name__ == '__main__':
    case_idx = '0025'
    img_path = f'data/WORD/imagesVal/word_{case_idx}.nii.gz'
    # img_path = 'data/WORD/imagesVal/word_0025.nii.gz'
    label_path = img_path.replace('imagesVal', 'labelsVal')

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()
    img = scale(img)
    cls_idx = 4
    slice_idx = 124
    use_conv = False
    alpha = 1.0
    palette = WORD_METAINFO['palette']
    img_slice = png_to_jpg(img[..., slice_idx], [255, 255, 255])
    mmcv.imwrite(img=img_slice, file_path=f'save_dirs/paper_vis/img_slice{slice_idx}.jpg')

    # pred = nib.load(
    #     'predictions/bkd_stage3_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96/20240608_141404/predictions/word_0025.nii.gz').get_fdata()
    pred = nib.load(
        f'predictions/unetmod_tiny_d8_300e_sgd_word_96x96x96/20240603_181227/predictions/word_{case_idx}.nii.gz').get_fdata()
    pred_slice = pred[..., slice_idx]
    label_slice = label[..., slice_idx]

    def get_edge_and_save(label_file, filename):
        label = nib.load(label_file).get_fdata()
        edge_mask = get_edges(label, cls_idx, use_conv)
        ret = cover(img_slice.copy(), edge_mask[..., slice_idx], [255, 0, 0], alpha).astype(np.uint8)
        mmcv.imwrite(
            img=ret,
            file_path=f'save_dirs/paper_vis/word_{case_idx}_{filename}_edge_slice{slice_idx}_cls{cls_idx}_use_conv{use_conv}.jpg')

    get_edge_and_save(
        f'predictions/unetmod_tiny_d8_300e_sgd_word_96x96x96/20240603_181227/predictions/word_{case_idx}.nii.gz',
        'student')
    get_edge_and_save(
        f'predictions/bkd_stage3_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96/20240608_141404/predictions/word_{case_idx}.nii.gz',
        'bkd')
    get_edge_and_save(
        f'predictions/multiscale_stage3_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96/20240608_142614/predictions/word_{case_idx}.nii.gz',
        'bkd+hd'
    )
    get_edge_and_save(
        f'predictions/multiscale_eta050_swinunetr_base_espnetv2_300e_sgd_word_96x96x96/20240606_161600/predictions/word_{case_idx}.nii.gz',
        'mskd+bkd+hd'
    )
    # # mask = np.expand_dims(pred == cls_idx, -1)
    # # mask = png_to_jpg(mask, palette[cls_idx])
    # edge_mask = get_edges(pred, cls_idx, use_conv)
    # # edge_mask_slice = png_to_jpg(edge_mask[..., slice_idx], [255, 0, 0])
    # # ret = cover(img_slice, edge_mask_slice).astype(np.uint8)
    # ret = cover(img_slice, edge_mask[..., slice_idx], [255, 0, 0], alpha).astype(np.uint8)
    # im = Image.fromarray(ret)
    # # im.save(
    # #     f'save_dirs/paper_vis/ret_slice{slice_idx}_use_conv{use_conv}.jpg',
    # #     quality=100, dpi=(2048, 2048))
    # mmcv.imwrite(img=ret, file_path=f'save_dirs/paper_vis/student_slice{slice_idx}_use_conv{use_conv}.jpg')

    edge_mask = get_edges(label, cls_idx, use_conv)
    # edge_mask_slice = png_to_jpg(edge_mask[..., slice_idx], [0, 255, 0])
    # ret = cover(ret, edge_mask_slice).astype(np.uint8)
    ret = cover(img_slice.copy(), edge_mask[..., slice_idx], [255, 0, 0], alpha).astype(np.uint8)
    # im = Image.fromarray(ret)
    # im.save(
    #     f'save_dirs/paper_vis/label_edge_slice{slice_idx}_use_conv{use_conv}.jpg',
    #     quality=100, dpi=(2048, 2048))
    mmcv.imwrite(
        img=ret,
        file_path=f'save_dirs/paper_vis/word_{case_idx}_label_edge_slice{slice_idx}_cls{cls_idx}_use_conv{use_conv}.jpg')
    # plt.imshow(ret)
    # plt.savefig(f'save_dirs/paper_vis/label_edge_slice{slice_idx}.svg')
    # plt.show()


