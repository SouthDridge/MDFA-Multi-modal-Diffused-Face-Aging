"""
This script encodes both original RGB images and 2D landmark heatmaps into 
the Latent Space using a pretrained Stable Diffusion VAE (Variational Autoencoder).

The transformation pipeline:
1. Input: 512x512 pixels (Image or Heatmap).
2. Processing: VAE Encoding (8x downsampling).
3. Output: 64x64x4 Latent Tensor.

This preprocessing is essential for training Latent Diffusion Models (LDM) 
as it allows the model to learn features in a computationally efficient 
compressed space.
"""

import os
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image

# Update these paths to your specific local directories
IMAGE_ROOT = "path/to/your/original_images" 
LDM_ROOT   = "path/to/your/point_heatmap_pt_files" 
SAVE_ROOT  = "path/to/your/output_latents" 

# Define subfolders for latent output
img_latent_folder = os.path.join(SAVE_ROOT, "img_latent")
ldm_latent_folder = os.path.join(SAVE_ROOT, "ldm_latent")

os.makedirs(img_latent_folder, exist_ok=True)
os.makedirs(ldm_latent_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
vae.to(device)
vae.eval()

# Preprocessing for RGB images: Resize -> Tensor -> Normalize to [-1, 1]
preprocess_img = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Utility Function
def extract_latent_from_tensor(img_tensor):
    """
    Encodes an image-like tensor into the VAE latent space.
    
    Args:
        img_tensor (Tensor): Shape [C, H, W], values typically in [0, 1].
    Returns:
        Tensor: Latent representation of shape [4, 64, 64].
    """
    img_tensor = img_tensor.float().to(device)
    
    # Expand single channel (grayscale heatmaps) to 3 channels (RGB) for VAE compatibility
    if img_tensor.ndim == 3 and img_tensor.shape[0] != 3:
        img_tensor = img_tensor.repeat(3, 1, 1)
        
    # Ensure the tensor is normalized to the [-1, 1] range required by the VAE
    if img_tensor.max() <= 1.0:
        img_tensor = img_tensor * 2.0 - 1.0
        
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension [1, 3, 512, 512]
    
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
        
    return latent.squeeze(0).cpu()  # Remove batch dimension and move to CPU

# Main Processing Logic
def process_images_folder(input_folder, save_folder):
    """Encodes standard image files (.jpg, .png) into latents."""
    image_files = sorted([
        f for f in os.listdir(input_folder) 
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    
    for file in tqdm(image_files, desc="Encoding Image Latents"):
        input_path = os.path.join(input_folder, file)
        save_path = os.path.join(save_folder, os.path.splitext(file)[0] + ".pt")
        
        if os.path.exists(save_path):
            continue
            
        img = Image.open(input_path).convert("RGB")
        img_tensor = preprocess_img(img)
        latent = extract_latent_from_tensor(img_tensor)
        torch.save(latent, save_path)

def process_ldm_folder(input_folder, save_folder):
    """Encodes pre-generated landmark heatmap tensors (.pt) into latents."""
    pt_files = sorted([
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(".pt")
    ])
    
    for file in tqdm(pt_files, desc="Encoding Heatmap Latents"):
        input_path = os.path.join(input_folder, file)
        
        name = os.path.splitext(file)[0].split("_")[0]
        save_path = os.path.join(save_folder, f"{name}.pt")
        
        if os.path.exists(save_path):
            continue

        data = torch.load(input_path, map_location='cpu')
        
        # Standardize data dimensions to [C, H, W] for the encoder
        if data.ndim == 2:
            data = data.unsqueeze(0) # [H, W] -> [1, H, W]
        elif data.ndim == 4:
            data = data[0]           # [1, 1, H, W] -> [1, H, W]

        latent = extract_latent_from_tensor(data)
        torch.save(latent, save_path)

# Execution
if __name__ == "__main__":
    process_images_folder(IMAGE_ROOT, img_latent_folder)
    process_ldm_folder(LDM_ROOT, ldm_latent_folder)
    print("Latent extraction completed successfully!")