import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerAdam(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerVanillaAdam(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(),
                         lr=self.initial_lr,
                         weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerVanillaAdam1en3(nnUNetTrainerVanillaAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerVanillaAdam3en4(nnUNetTrainerVanillaAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4

class nnUNetTrainerVanillaRAdam3en4(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug=debug)
        self.initial_lr = 3e-4

    def configure_optimizers(self):
        optimizer = RAdam(self.network.parameters(),
                         lr=self.initial_lr,
                         weight_decay=self.weight_decay)
        # lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=5e-6,
            last_epoch=-1
        )
        return optimizer, lr_scheduler


class nnUNetTrainerVanillaRAdam3en4OneCycleLR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, debug=debug)
        self.initial_lr = 3e-4
        self.num_epochs = 300
    def configure_optimizers(self):
        optimizer = RAdam(self.network.parameters(),
                         lr=self.initial_lr,
                         weight_decay=self.weight_decay)
        kwargs = {
            'max_lr': self.initial_lr,
            'total_steps': self.num_epochs,
            'steps_per_epoch': 1,
            'epochs': self.num_epochs,
            'pct_start': 0.1,
            'final_div_factor': 1e8,
            'anneal_strategy': 'cos', #'cos' | 'linear'
            # 'three_phase': False, # True
        }
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        return optimizer, lr_scheduler


class nnUNetTrainerAdam1en3(nnUNetTrainerAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerAdam3en4(nnUNetTrainerAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4
