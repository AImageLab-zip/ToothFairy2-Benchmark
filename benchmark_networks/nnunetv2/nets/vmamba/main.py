from vmamba import Backbone_VSSM
from seg_vmamba import SegVMamba
import torch

print("Inizio...")
backbone_model = Backbone_VSSM(
        in_chans=3,
        num_classes=2,
        # Copied from configs files of segmentation folder on github
        depths=[2, 2, 15, 2],
        dims=128,
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",  # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
        out_indices=(0,1,2,3),
        norm_layer="ln2d"
    )

# TRY THE BACKBONE ONLY
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")
backbone_model = backbone_model.to(device)
x = torch.randn(1, 3, 256, 256).to(device)
y = torch.randn(56, 16, 263, 222).to(device)

print("Inference backbone model...")
out_x = backbone_model(x)
out_y = backbone_model(y)
with open("output_4outidx.txt", "w") as file:
    file.write(f"Shape of x: {x.shape}\n")
    file.write(f"Length of out_x: {len(out_x)}\n")
    for idx, _x in enumerate(out_x):
        file.write(f"Shape of out_x[{idx}]: {out_x[idx].shape}\n")

    file.write(f"Shape of y: {y.shape}\n")
    file.write(f"Length of out_y: {len(out_y)}\n")
    for idx, _y in enumerate(out_y):
        file.write(f"Shape of out_y[{idx}]: {out_y[idx].shape}\n")

# TRY ALL THE SEGMENTATION MODEL
seg_model = SegVMamba(num_classes=2, backbone=backbone_model, fpn_out=128)
seg_model = seg_model.to(device)

print("Inference segmentation model...")
#seg_out_x = seg_model(x)
seg_out_y = seg_model(y)
with open("output_segmodel.txt", "w") as file:
    #file.write(f"Shape of x: {x.shape}\n")
    #file.write(f"Length of out_x: {seg_out_x.shape}\n")

    file.write(f"Shape of y: {y.shape}\n")
    file.write(f"Length of out_y: {seg_out_y.shape}\n")

