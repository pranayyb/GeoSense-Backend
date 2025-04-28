import os
import torch

# Device
DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
)

# Model config
MODEL_TYPE = "vit_h"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Paths
HOME = os.getcwd()
WEIGHTS_DIR = os.path.join(HOME, "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")

# Other constants
SCALE_NORMALIZER = 1.5136
ACRES_CONVERSION = 4046.856422
