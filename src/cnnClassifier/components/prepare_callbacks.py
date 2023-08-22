from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

import os
import time
import torch 
from tensorboardX import SummaryWriter



class PrepareCallback():
    def __init__(self, config:PrepareCallbacksConfig):
        self.config = config
        self.writer = None

    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,f"tb_logs_at_{timestamp}"
        )
        return SummaryWriter(tb_running_log_dir)

    def _create_ckpt_callbacks(self, model, optimizer, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            self.config.checkpoint_model_filepath
        )

    def get_tb_ckpt_callbacks(self, model, optimizer, epoch, loss):
        return [self._create_tb_callbacks(), self._create_ckpt_callbacks(model, optimizer, epoch, loss)()]