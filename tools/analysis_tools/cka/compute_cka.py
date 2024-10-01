from cka import CKACalculator
import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from copy import deepcopy
from seg.datasets.get_dataloader import BTCV_loader
from configs._base_.datasets.synapse import dataloader_cfg

num_epochs = 1

model1_name = 'swin_unetr_b'
model1_config = 'configs/swin_unetr/swinunetr_base_5000e_synapse.py'
model1_ckpt = 'ckpts/swin_unetr.base_5000ep_f48_lr2e-4_pretrained_mmengine.pth'

# model1_config = 'configs/unet/unetmod_base_d8_1000e_sgd_synapse_96x96x96.py'
# model1_ckpt = 'ckpts/unetmod_base_d8_1000e_sgd_synapse_96x96x96/best_Dice_81-69_epoch_800.pth'

# model1_name = 'uxnet'
# model1_config = 'configs/uxnet/uxnet_b1_2000e_adamw_noscheduler_synapse_96x96x96.py'
# model1_ckpt = 'ckpts/uxnet_b1_2000e_adamw_noscheduler_synapse_96x96x96/best_Dice_80-89_epoch_1600.pth'

model2_name = 'unet_tiny'
model2_config = 'configs/unet/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96.py'
model2_ckpt = 'ckpts/unetmod_tiny_d8_1000e_sgd_synapse_96x96x96/best_Dice_66-21_epoch_1000.pth'

# model2_name = 'swin_unetr_t'
# model2_config = 'configs/swin_unetr/swinunetr_tiny_5000e_sgd_synapse_96x96x96.py'
# model2_ckpt = 'work_dirs/swinunetr_tiny_5000e_sgd_synapse_96x96x96/5-run_20240430_123451/run0/best_Dice_80-66_epoch_4750.pth'

model1 = MODELS.build(Config.fromfile(model1_config).model).cuda()
model1.eval()
# model2 = deepcopy(model1)
# model2_name = model1_name
model2 = MODELS.build(Config.fromfile(model2_config).model).cuda()
model2.eval()

load_checkpoint(model1, model1_ckpt)
# load_checkpoint(model2, model1_ckpt)
load_checkpoint(model2, model2_ckpt)

dataloader_cfg = Config(dataloader_cfg)
# dataloader_cfg.use_normal_dataset = True
dataloader_cfg.workers = 1
dataloader = BTCV_loader(dataloader_cfg, test_mode=False, save=False)[0]
dataloader.pin_memory = False
hook_layer_types = (nn.InstanceNorm3d, nn.LeakyReLU, nn.PReLU, nn.ConvTranspose3d, nn.Conv3d)
calculator = CKACalculator(
    model1=model1,
    model2=model2,
    dataloader=dataloader,
    num_epochs=num_epochs,
    # hook_fn='avgpool',
    # hook_layer_types=hook_layer_types,
    # hook_layer_types=(nn.InstanceNorm3d, ),
    # hook_layer_types=(nn.LeakyReLU, nn.PReLU),
    hook_layer_types=(nn.Conv3d,
                      nn.ConvTranspose3d
                      ),
    hook_layer_names=('conv3.conv', '')
)
cka_output = calculator.calculate_cka_matrix()
print(f"CKA output size: {cka_output.size()}")

print("model1 layer names:")
for i, name in enumerate(calculator.module_names_X):
    print(f"Layer {i}: \t{name}")

print("model2 layer names:")
for i, name in enumerate(calculator.module_names_Y):
    print(f"Layer {i}: \t{name}")

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
plt.imshow(cka_output.cpu().numpy(), cmap='inferno', interpolation='nearest', aspect='auto')
plt.gca().invert_yaxis()
# plt.show()
import time
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
plt.savefig(f'save_dirs/cka/{model1_name}-{model2_name}-{num_epochs}e-{"Conv3d+trans"}-{timestamp}.jpg')