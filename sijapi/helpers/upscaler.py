from aura_sr import AuraSR
from PIL import Image
import torch
import os

# Set environment variables for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Initialize device as CPU for default
device = torch.device('cpu')

# Check if MPS is available
if torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        device = torch.device('mps:0')

# Overwrite the default CUDA device with MPS
torch.cuda.default_stream = device

aura_sr = AuraSR.from_pretrained("fal-ai/AuraSR").to(device)

def load_image_from_path(file_path):
    return Image.open(file_path)

def upscale_and_save(original_path):
    original_image = load_image_from_path(original_path)
    upscaled_image = aura_sr.upscale_4x(original_image)
    upscaled_image.save(original_path)

# Insert your image path
upscale_and_save("/Users/sij/workshop/sijapi/sijapi/testbed/API__00482_ 2.png")
