from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans

from calflops import calculate_flops
import time
from fvcore.nn import FlopCountAnalysis



class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), debug=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        #print("UMambaBot: {}".format(model))
        #input_synapse = torch.rand((2, 1, 48, 192, 192))
        input_brain = torch.rand((2, 4, 128, 128, 128))
        #input_acdc = torch.rand((4, 1, 14, 256, 224))

        total_params = sum(
        param.numel() for param in model.parameters()
        )
        print("UMambaBot Params: {}".format(total_params))

        #flops = FlopCountAnalysis(model, input_shape_brain)
        #print("GFlops {}".format(flops.total() / 1e9))

        #model = model.to("cuda")
        args = [input_brain.to("cuda")]
        flops, macs, params = calculate_flops(model=model,
                                               #input_shape=tuple(input_brain.shape),
                                               args=args,
                                               output_as_string=True,
                                               output_precision=4)
        print("UMbambaBot Brain Tumor FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

        time.sleep(60)

        return model

