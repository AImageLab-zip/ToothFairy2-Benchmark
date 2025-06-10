from torch.optim.lr_scheduler import CosineAnnealingLR
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from torch.optim import AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from nnunetv2.nets.vmamba.seg_vmamba import get_seg_vmamba_from_plans


class nnUNetTrainerVMamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300
        self.weight_decay = 0.05
        self.enable_deep_supervision = False

    def configure_optimizers(self):
        # hyperparameters obtained by https://github.com/JCruan519/VM-UNet/blob/main/configs/config_setting_synapse.py
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False)

        '''lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=5e-6,
            last_epoch=-1
        )'''
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_seg_vmamba_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            raise NotImplementedError("Only 2D models are supported")
        else:
            raise NotImplementedError("Only 2D models are supported")

        
        print("VMamba: {}".format(model))

        return model

