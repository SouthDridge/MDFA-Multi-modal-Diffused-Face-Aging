"""
This script performs inference using the trained Multi-Level Cross-Attention Adapter.
It uses a StableDiffusionImg2ImgPipeline to generate age-transformed faces 
while maintaining structural control via precomputed landmark latents.
"""

import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
import torch.nn.functional as F
import torch.nn as nn

# ---------------- Multi-Level Cross Attention (Same as training) ----------------
class MultiLevelCrossAttention(nn.Module):
    def __init__(self, in_ch=4, target_channels=[320,640,1280,1280], num_heads=[8,8,8,8]):
        super().__init__()
        self.num_levels = len(target_channels)
        self.blocks = nn.ModuleList()

        for out_ch, heads in zip(target_channels, num_heads):
            self.blocks.append(nn.ModuleDict({
                "q_proj": nn.Conv2d(in_ch, out_ch, 1),
                "k_proj": nn.Conv2d(in_ch, out_ch, 1),
                "v_proj": nn.Conv2d(in_ch, out_ch, 1),
                "attn": nn.MultiheadAttention(out_ch, heads, batch_first=False),
                "out_proj": nn.Conv2d(out_ch, out_ch, 1)
            }))

    def forward(self, face_latent, lm_latent):
        B, _, H, W = face_latent.shape
        outs = []
        for i, blk in enumerate(self.blocks):
            scale = 2 ** i
            f = F.avg_pool2d(face_latent, scale) if scale > 1 else face_latent
            l = F.avg_pool2d(lm_latent, scale) if scale > 1 else lm_latent

            q = blk["q_proj"](f).flatten(2).permute(2, 0, 1)
            k = blk["k_proj"](l).flatten(2).permute(2, 0, 1)
            v = blk["v_proj"](l).flatten(2).permute(2, 0, 1)
            
            attn_out, _ = blk["attn"](q, k, v)
            attn_out = attn_out.permute(1, 2, 0).reshape(B, -1, f.shape[2], f.shape[3])
            outs.append(blk["out_proj"](attn_out))
        return outs

# ---------------- Latent Adapter (Same as training) ----------------
class LatentAdapter(nn.Module):
    def __init__(self, latent_ch=320, base_ch=64, target_channels=[320,640,1280,1280]):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU()
        )
        self.proj_blocks = nn.ModuleList()
        for i, tgt in enumerate(target_channels):
            in_ch = base_ch * (2**(2*i))
            self.proj_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, base_ch, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_ch, tgt, 1)
            ))

    def forward(self, fused_list):
        x = self.in_proj(fused_list[0])
        outs = []
        for i, blk in enumerate(self.proj_blocks):
            factor = 2**i
            y = x if factor == 1 else F.pixel_unshuffle(x, downscale_factor=factor)
            outs.append(blk(y))
        return outs

# Wrapper to inject precomputed adapter features into UNet
class PrecomputedAdapter:
    def __init__(self, feats):
        self.feats = feats
    def __call__(self, *args, **kwargs):
        return self.feats
    def to(self, device, dtype=torch.float16):
        self.feats = [f.to(device, dtype) for f in self.feats]
        return self

# Main Inference
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_latent_dir", required=True)
    parser.add_argument("--landmark_latent_dir", required=True)
    parser.add_argument("--input_image_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--ckpt_root", required=True)  # Dir containing age_XX/best_epoch.pt
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--adapter_strength", type=float, default=0.7)
    parser.add_argument("--img_strength", type=float, default=0.3)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16

    # Load SD v1.5 Img2Img Pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    pipe.enable_xformers_memory_efficient_attention()

    # Target age groups
    ages = [65] 
    prompts = [f"a realistic face photo of a {a}-year-old person" for a in ages]
    image_list = sorted([f for f in os.listdir(args.input_image_dir) if f.lower().endswith((".png",".jpg",".jpeg"))])

    for age, prompt in zip(ages, prompts):
        ckpt = os.path.join(args.ckpt_root, f"age_{age}", "best_epoch.pt")
        if not os.path.exists(ckpt):
            print(f"Skipping Age {age}: Checkpoint not found at {ckpt}")
            continue

        age_out = os.path.join(args.output_root, f"age_{age}")
        os.makedirs(age_out, exist_ok=True)

        # Initialize and load adapter modules
        cross = MultiLevelCrossAttention().to(device, dtype)
        adapter = LatentAdapter().to(device, dtype)
        weights = torch.load(ckpt, map_location=device)
        cross.load_state_dict(weights["cross_attn"], strict=False)
        adapter.load_state_dict(weights["adapter"], strict=False)
        cross.eval(); adapter.eval()

        for name in tqdm(image_list, desc=f"Processing Age {age}"):
            base = os.path.splitext(name)[0]
            f_path = os.path.join(args.face_latent_dir, base + ".pt")
            l_path = os.path.join(args.landmark_latent_dir, base + ".pt")
            
            if not os.path.exists(f_path) or not os.path.exists(l_path):
                continue

            # Prepare latents
            f_lat = torch.load(f_path).unsqueeze(0).to(device, dtype)
            l_lat = torch.load(l_path).unsqueeze(0).to(device, dtype)
            
            with torch.no_grad():
                fused = cross(f_lat, l_lat)
                feats = adapter(fused)

            # Inject adapter features
            pipe.adapter = PrecomputedAdapter(feats).to(device, dtype)

            # Prepare initial image for Img2Img
            init_img = Image.open(os.path.join(args.input_image_dir, name)).convert("RGB").resize((args.size, args.size))
            
            # Generate
            output = pipe(
                prompt=prompt,
                image=init_img,
                strength=args.img_strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                adapter_conditioning_scale=args.adapter_strength,
                negative_prompt="cartoon, anime, artifact, distortion, low quality"
            ).images[0]

            output.save(os.path.join(age_out, name))

    print("Inference Task Completed.")

if __name__ == "__main__":
    main()