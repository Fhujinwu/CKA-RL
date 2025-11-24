import torch
import torch.nn as nn
import os
from .shared_arch import shared
from .mask_modules import MultitaskMaskLinear, set_model_task, NEW_MASK_LINEAR_COMB, consolidate_mask


class MaskNetAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, num_tasks):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.num_tasks = num_tasks
        
        self.fc = shared(input_dim=obs_dim)

        # will be created when calling `reset_heads`
        self.fc_mean = None
        self.fc_logstd = None
        self.reset_heads()

    def reset_heads(self):
        self.fc_mean = MultitaskMaskLinear(256, self.act_dim, \
            discrete=True, num_tasks=self.num_tasks, new_mask_type=NEW_MASK_LINEAR_COMB)
        self.fc_logstd = MultitaskMaskLinear(256, self.act_dim, \
            discrete=True, num_tasks=self.num_tasks, new_mask_type=NEW_MASK_LINEAR_COMB)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    def load(dirname, map_location=None):
        model = torch.load(f"{dirname}/model.pt", map_location=map_location)
        return model

    def set_task(self, task, new_task):
        set_model_task(self, task, new_task=new_task, verbose=True)
        
    def consolidate_mask(self):
        consolidate_mask(self)