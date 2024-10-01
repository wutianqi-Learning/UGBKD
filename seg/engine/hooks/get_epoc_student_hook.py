# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from seg.registry import HOOKS
from mmengine.logging import MMLogger, print_log
@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

        runner.model.distiller.distill_losses.loss_cwd.epoch = runner.epoch+1
        # runner.model.teacher.decode_head.loss_decode[1].epoch = runner.epoch+1
        # runner.model.auxiliary_head.loss_decode[1].iter = runner.iter



