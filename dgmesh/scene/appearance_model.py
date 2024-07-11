import os
import torch
from utils.time_utils import AppearanceNetwork
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class AppearanceModel:
    def __init__(self, is_blender=False, is_6dof=False):
        self.appearance_net = AppearanceNetwork(is_blender=is_blender).cuda()
        self.optimizer = None

    def step(self, xyz, time_emb):
        return self.appearance_net(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.appearance_net.parameters()),
             'lr': training_args.apperance_lr_init,
             "name": "appearance"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.apperance_lr_init,
                                                       lr_final=training_args.apperance_lr_final,
                                                       lr_delay_mult=training_args.apperance_lr_delay_mult,
                                                       max_steps=training_args.apperance_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "appearance/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.appearance_net.state_dict(), os.path.join(out_weights_path, 'appearance.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "appearance"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "appearance/iteration_{}/appearance.pth".format(loaded_iter))
        self.appearance_net.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr