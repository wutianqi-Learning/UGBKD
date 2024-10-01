"""
Tool to compute Centered Kernel Alignment (CKA) in PyTorch w/ GPU (single or multi).

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable, Type, Union, TYPE_CHECKING, List

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from hook_manager import HookManager, _HOOK_LAYER_TYPES
from metrics import AccumTensor
from cka import CKACalculator
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from copy import deepcopy
from seg.datasets.get_dataloader import BTCV_loader
from configs._base_.datasets.synapse import dataloader_cfg

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class Calculator:
    def __init__(self,
                 # model1: nn.Module, model2: nn.Module,
                 # dataloader: DataLoader,
                 hook_fn: Optional[Union[str, Callable]] = None,
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES, num_epochs: int = 10,
                 hook_layer_names: Tuple[str, ...] = (),
                 group_size: int = 512, epsilon: float = 1e-4, is_main_process: bool = True) -> None:
        """
        Class to extract intermediate features and calculate CKA Matrix.
        :param model1: model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param model2: second model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param dataloader: Torch DataLoader for dataloading. Assumes first return value contains input images.
        :param hook_fn: Optional - Hook function or hook name string for the HookManager. Options: [flatten, avgpool]. Default: flatten
        :param hook_layer_types: Types of layers (modules) to add hooks to.
        :param num_epochs: Number of epochs for cka_batch. Default: 10
        :param group_size: group_size for GPU acceleration. Default: 512
        :param epsilon: Small multiplicative value for HSIC. Default: 1e-4
        :param is_main_process: is current instance main process. Default: True
        """
        # self.model1 = model1
        # self.model2 = model2
        # self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process

        # self.model1.eval()
        # self.model2.eval()
        # self.hook_manager1 = HookManager(self.model1, hook_fn, hook_layer_types, hook_layer_names, calculate_gram=True)
        # self.hook_manager2 = HookManager(self.model2, hook_fn, hook_layer_types, hook_layer_names, calculate_gram=True)
        self.module_names_X = None
        self.module_names_Y = None
        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None

        # Metrics to track
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:
        curr_hsic_matrix = None
        curr_self_hsic_x = None
        curr_self_hsic_y = None
        for epoch in range(self.num_epochs):
            loader = tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process)
            # for it, (imgs, *_) in enumerate(loader):
            for idx, data_batch in enumerate(loader):
                # imgs = imgs.cuda(non_blocking=True)
                data = self.model1.data_preprocessor(data_batch, True)
                self.model1._run_forward(data, mode='loss')  # type: ignore
                self.model2._run_forward(data, mode='loss')  # type: ignore
                all_layer_X, all_layer_Y = self.extract_layer_list_from_hook_manager()

                # Initialize values on first loop
                if self.num_layers_X is None:
                    curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y = self._init_values(all_layer_X, all_layer_Y)

                # Get self HSIC values --> HSIC(K, K), HSIC(L, L)
                self._calculate_self_hsic(all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y)

                # Get cross HSIC values --> HSIC(K, L)
                self._calculate_cross_hsic(all_layer_X, all_layer_Y, curr_hsic_matrix)

                self.hook_manager1.clear_features()
                self.hook_manager2.clear_features()
                curr_hsic_matrix.fill_(0)
                curr_self_hsic_x.fill_(0)
                curr_self_hsic_y.fill_(0)
                # break

        # Update values across GPUs
        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()
        self.cka_matrix = hsic_matrix.reshape(self.num_layers_Y, self.num_layers_X) / torch.sqrt(hsic_x * hsic_y)
        # print(self.cka_matrix.diagonal())
        # self.cka_matrix = self.cka_matrix.flip(0)
        return self.cka_matrix

    def extract_layer_list_from_hook_manager(self) -> Tuple[List, List]:
        all_layer_X, all_layer_Y = self.hook_manager1.get_features(), self.hook_manager2.get_features()
        return all_layer_X, all_layer_Y

    def hsic1(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Batched version of HSIC.
        :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :return: HSIC tensor, Size = (B)
        '''
        assert K.size() == L.size()
        assert K.dim() == 3
        K = K.clone()
        L = L.clone()
        n = K.size(1)

        # K, L --> K~, L~ by setting diagonals to zero
        K.diagonal(dim1=-1, dim2=-2).fill_(0)
        L.diagonal(dim1=-1, dim2=-2).fill_(0)

        KL = torch.bmm(K, L)
        trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
        middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
        middle_term /= (n - 1) * (n - 2)
        right_term = KL.sum((-1, -2), keepdim=True)
        right_term *= 2 / (n - 2)
        main_term = trace_KL + middle_term - right_term
        hsic = main_term / (n ** 2 - 3 * n)
        return hsic.squeeze(-1).squeeze(-1)

    def reset(self) -> None:
        # Set values to none, clear feature and hooks
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None
        self.hook_manager1.clear_all()
        self.hook_manager2.clear_all()

    def init_values(self, all_layer_X, all_layer_Y):
        self.num_layers_X = len(all_layer_X)
        self.num_layers_Y = len(all_layer_Y)
        # self.module_names_X = self.hook_manager1.get_module_names()
        # self.module_names_Y = self.hook_manager2.get_module_names()
        self.num_elements = self.num_layers_Y * self.num_layers_X
        curr_hsic_matrix = torch.zeros(self.num_elements).cuda()
        curr_self_hsic_x = torch.zeros(1, self.num_layers_X).cuda()
        curr_self_hsic_y = torch.zeros(self.num_layers_Y, 1).cuda()
        self.hsic_matrix = AccumTensor(torch.zeros_like(curr_hsic_matrix)).cuda()
        self.self_hsic_x = AccumTensor(torch.zeros_like(curr_self_hsic_x)).cuda()
        self.self_hsic_y = AccumTensor(torch.zeros_like(curr_self_hsic_y)).cuda()
        return curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y

    def calculate_self_hsic(self, all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y):
        for start_idx in range(0, self.num_layers_X, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_X)
            K = torch.stack([all_layer_X[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_x[0, start_idx:end_idx] += self.hsic1(K, K) * self.epsilon
        for start_idx in range(0, self.num_layers_Y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_Y)
            L = torch.stack([all_layer_Y[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_y[start_idx:end_idx, 0] += self.hsic1(L, L) * self.epsilon

        self.self_hsic_x.update(curr_self_hsic_x)
        self.self_hsic_y.update(curr_self_hsic_y)

    def calculate_cross_hsic(self, all_layer_X, all_layer_Y, curr_hsic_matrix):
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            K = torch.stack([all_layer_X[i % self.num_layers_X] for i in range(start_idx, end_idx)], dim=0)
            L = torch.stack([all_layer_Y[j // self.num_layers_X] for j in range(start_idx, end_idx)], dim=0)
            curr_hsic_matrix[start_idx:end_idx] += self.hsic1(K, L) * self.epsilon
        self.hsic_matrix.update(curr_hsic_matrix)


def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())


if __name__ == '__main__':
    # model1_name = 'swin_unetr_b'
    # model1_config = 'configs/swin_unetr/swinunetr_base_5000e_synapse.py'
    # model1_ckpt = 'ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'

    # model1_config = 'configs/unet/unetmod_base_d8_1000e_sgd_synapse_96x96x96.py'
    # model1_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'

    model1_name = 'uxnet'
    model1_config = 'configs/uxnet/uxnet_b1_2000e_adamw_noscheduler_synapse_96x96x96.py'
    model1_ckpt = 'ckpts/uxnet_b1_2000e_adamw_noscheduler_synapse_96x96x96/best_Dice_80-89_epoch_1600.pth'

    # model2_name = 'unet_tiny'
    # model2_config = 'configs/unet/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96.py'
    # model2_ckpt = 'ckpts/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96/best_Dice_66-21_epoch_1000.pth'

    # model2_name = 'swin_unetr_t'
    # model2_config = 'configs/swin_unetr/swinunetr_tiny_5000e_sgd_synapse_96x96x96.py'
    # model2_ckpt = 'work_dirs/swinunetr_tiny_5000e_sgd_synapse_96x96x96/5-run_20240430_123451/run0/best_Dice_80-66_epoch_4750.pth'

    model1 = MODELS.build(Config.fromfile(model1_config).model).cuda()
    model1.eval()
    model2 = deepcopy(model1)
    model2_name = model1_name
    # model2 = MODELS.build(Config.fromfile(model2_config).model).cuda()
    model2.eval()

    load_checkpoint(model1, model1_ckpt)
    load_checkpoint(model2, model1_ckpt)
    # load_checkpoint(model2, model2_ckpt)

    dataloader_cfg = Config(dataloader_cfg)
    # dataloader_cfg.use_normal_dataset = True
    dataloader_cfg.workers = 1
    dataloader = BTCV_loader(dataloader_cfg, test_mode=False, save=False)[0]
    dataloader.pin_memory = False



    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = (7, 7)
    plt.imshow(cka_matrix.cpu().numpy(), cmap='inferno', interpolation='nearest', aspect='auto')
    plt.gca().invert_yaxis()
    plt.show()
    print(1)
