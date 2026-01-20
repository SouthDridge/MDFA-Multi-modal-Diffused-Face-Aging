"""
This script trains a Latent-level Multi-scale Cross-Attention Adapter for Stable Diffusion.
It takes paired face latents and landmark heatmaps (both in VAE latent space) to 
learn structural control over face generation.
"""

import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from diffusers import StableDiffusionPipeline, DDPMScheduler

# Dataset
class LatentDataset(Dataset):
    def __init__(self, face_latent_dir, landmark_latent_dir, prompt="a portrait of a person"):
        face_paths = sorted(glob(os.path.join(face_latent_dir, "*.pt")))
        landmark_paths = sorted(glob(os.path.join(landmark_latent_dir, "*.pt")))
        
        face_bases = {os.path.splitext(os.path.basename(p))[0]: p for p in face_paths}
        landmark_bases = {os.path.splitext(os.path.basename(p))[0]: p for p in landmark_paths}
        
        # Ensure only paired files are kept
        self.pairs = [(f_path, landmark_bases[base]) for base, f_path in face_bases.items() if base in landmark_bases]
        
        if len(self.pairs) == 0:
            raise RuntimeError("No valid paired latents found. Please check your directory paths.")
        self.prompt = prompt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        face_path, landmark_path = self.pairs[idx]
        face_latent = torch.load(face_path).float()
        landmark_latent = torch.load(landmark_path).float()
        return {"face_latent": face_latent, "landmark_latent": landmark_latent, "prompt": self.prompt}


def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return None if len(batch) == 0 else default_collate(batch)


# Adapter (pixel_unshuffle)
class LatentAdapter(nn.Module):
    """
    Adapter supporting either a single high-res latent or multi-scale latent list.
    Uses pixel_unshuffle to match UNet down-sampling stages.
    """
    def __init__(self, latent_ch=320, base_ch=64, target_channels=[320, 640, 1280, 1280]):
        super().__init__()
        self.latent_ch = latent_ch
        self.base_ch = base_ch
        self.target_channels = target_channels
        self.num_levels = len(target_channels)

        # Projection for merging multi-scale inputs
        self.merge_projs = nn.ModuleList([
            nn.Conv2d(tgt, base_ch, 1) for tgt in target_channels
        ])

        # Initial processing block
        self.in_proj = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU()
        )

        # Per-level projection blocks
        self.proj_blocks = nn.ModuleList()
        for i, tgt in enumerate(target_channels):
            # Channels increase by factor^2 after pixel_unshuffle
            in_ch = base_ch * (2 ** (2 * i))
            blk = nn.Sequential(
                nn.Conv2d(in_ch, base_ch, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_ch, tgt, 1)
            )
            self.proj_blocks.append(blk)

    def forward(self, latent):
        if isinstance(latent, (list, tuple)):
            highest = latent[0]
            B, C, H, W = highest.shape
            merged = torch.zeros((B, self.base_ch, H, W), device=highest.device, dtype=highest.dtype)
            for i, l in enumerate(latent):
                proj = self.merge_projs[i](l)
                if proj.shape[-2:] != (H, W):
                    proj = F.interpolate(proj, size=(H, W), mode='bilinear', align_corners=False)
                merged = merged + proj
            x = self.in_proj(merged)
        else:
            x = self.in_proj(latent)

        outs = []
        for i, blk in enumerate(self.proj_blocks):
            factor = 2 ** i
            if factor == 1:
                y = x
            else:
                y = F.pixel_unshuffle(x, downscale_factor=factor)
            outs.append(blk(y))
        return outs


# Multi-level Cross-Attention
class MultiLevelCrossAttention(nn.Module):
    """
    Fuses face features and landmark features at multiple spatial scales.
    """
    def __init__(self, in_ch=4, target_channels=[320, 640, 1280, 1280], num_heads=None):
        super().__init__()
        self.target_channels = target_channels
        self.num_levels = len(target_channels)
        if num_heads is None:
            num_heads = [8] * self.num_levels
        self.num_heads = num_heads

        self.blocks = nn.ModuleList()
        for i in range(self.num_levels):
            out_ch = target_channels[i]
            heads = num_heads[i]
            self.blocks.append(nn.ModuleDict({
                "q_proj": nn.Conv2d(in_ch, out_ch, 1),
                "k_proj": nn.Conv2d(in_ch, out_ch, 1),
                "v_proj": nn.Conv2d(in_ch, out_ch, 1),
                "attn": nn.MultiheadAttention(embed_dim=out_ch, num_heads=heads, batch_first=False),
                "out_proj": nn.Conv2d(out_ch, out_ch, 1)
            }))

    def forward(self, face_latent, lm_latent):
        B, _, H, W = face_latent.shape
        outs = []
        for i, blk in enumerate(self.blocks):
            scale = 2 ** i
            f_down = F.avg_pool2d(face_latent, scale) if scale > 1 else face_latent
            l_down = F.avg_pool2d(lm_latent, scale) if scale > 1 else lm_latent
            
            q = blk["q_proj"](f_down).flatten(2).permute(2, 0, 1)
            k = blk["k_proj"](l_down).flatten(2).permute(2, 0, 1)
            v = blk["v_proj"](l_down).flatten(2).permute(2, 0, 1)
            
            attn_out, _ = blk["attn"](q, k, v)
            attn_out = attn_out.permute(1, 2, 0).reshape(B, -1, f_down.shape[2], f_down.shape[3])
            outs.append(blk["out_proj"](attn_out))
        return outs


# Hook for UNet
def add_adapter_to_unet(unet, adapter):
    handles = []
    down_blocks = [m for m in unet.down_blocks]
    adapter_module = adapter.module if isinstance(adapter, nn.parallel.DistributedDataParallel) else adapter

    target_len = len(adapter_module.target_channels)
    assert len(down_blocks) >= target_len

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            global CURRENT_ADAPTER_OUTPUTS
            if 'CURRENT_ADAPTER_OUTPUTS' in globals() and CURRENT_ADAPTER_OUTPUTS is not None:
                a = CURRENT_ADAPTER_OUTPUTS[idx]
                # Adjust spatial resolution and channels to match UNet internal layer
                if a.shape[-2:] != h.shape[-2:]:
                    a = F.interpolate(a, size=h.shape[-2:], mode='bilinear', align_corners=False)
                if a.shape[1] != h.shape[1]:
                    proj = nn.Conv2d(a.shape[1], h.shape[1], 1).to(a.device)
                    with torch.no_grad():
                        return h + proj(a)
                else:
                    return h + a
            return out
        return hook

    for i in range(target_len):
        module = down_blocks[i].resnets[-1] if hasattr(down_blocks[i], "resnets") else down_blocks[i]
        handles.append(module.register_forward_hook(make_hook(i)))
    return handles


# Training
def train(args):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Stable Diffusion Setup
    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipe.to(device)
    vae = pipe.vae.eval()
    text_encoder = pipe.text_encoder.eval()
    tokenizer = pipe.tokenizer
    unet = pipe.unet.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Freeze base model parameters
    for p in vae.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False
    for p in unet.parameters(): p.requires_grad = False

    # Dataloader
    dataset = LatentDataset(args.face_latent_dir, args.landmark_latent_dir, prompt=args.prompt)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_none
    )

    # Adapter Models
    adapter = LatentAdapter(
        latent_ch=args.adapter_target_channels[0],
        base_ch=args.adapter_base_ch,
        target_channels=args.adapter_target_channels
    ).to(device)
    
    cross_attn = MultiLevelCrossAttention(
        in_ch=4, target_channels=args.adapter_target_channels,
        num_heads=args.adapter_num_heads
    ).to(device)

    adapter = nn.parallel.DistributedDataParallel(adapter, device_ids=[local_rank])
    cross_attn = nn.parallel.DistributedDataParallel(cross_attn, device_ids=[local_rank])

    optimizer = torch.optim.AdamW([
        {"params": adapter.parameters(), "lr": args.lr_adapter},
        {"params": cross_attn.parameters(), "lr": args.lr_adapter}
    ], weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    handles = add_adapter_to_unet(unet, adapter)

    # State Tracking
    start_epoch = 0
    best_loss = float("inf")
    loss_history, ema_history = [], []
    ema_loss = None

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        if os.path.exists(ckpt_path):
            map_location = {"cuda:%d" % 0: f"cuda:{local_rank}"}
            ckpt = torch.load(ckpt_path, map_location=map_location)
            adapter.module.load_state_dict(ckpt["adapter"])
            cross_attn.module.load_state_dict(ckpt["cross_attn"])
            if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt: scaler.load_state_dict(ckpt["scaler"])
            if "epoch" in ckpt: start_epoch = ckpt["epoch"] + 1
            if "best_loss" in ckpt: best_loss = ckpt["best_loss"]
            print(f"[Rank {local_rank}]  Resumed from {ckpt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.txt")
    
    global CURRENT_ADAPTER_OUTPUTS
    CURRENT_ADAPTER_OUTPUTS = None

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        # Basic learning rate scheduler
        lr = 1e-4 if epoch < 10 else 1e-5 if epoch < 20 else 1e-6
        for g in optimizer.param_groups: g["lr"] = lr
        
        if local_rank == 0:
            print(f"Epoch {epoch} | LR: {lr}")

        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        epoch_losses = []

        for batch in pbar:
            if batch is None: continue
            face_latent = batch["face_latent"].to(device, non_blocking=True)
            lm_latent = batch["landmark_latent"].to(device, non_blocking=True)
            prompts = batch["prompt"]

            # Standard Diffusion Noise Process
            B = face_latent.shape[0]
            noise = torch.randn_like(face_latent)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
            noisy_latents = scheduler.add_noise(face_latent, noise, timesteps)

            # Text Encoding
            text_inputs = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_embeds = text_encoder(**text_inputs).last_hidden_state

            # Forward Pass with Adapter
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                fused_latents = cross_attn(face_latent, lm_latent)
                adapter_outs = adapter(fused_latents)
                CURRENT_ADAPTER_OUTPUTS = adapter_outs

                unet_out = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds)
                pred_noise = unet_out.sample if hasattr(unet_out, "sample") else unet_out
                loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # Logging
            if local_rank == 0:
                loss_item = loss.item()
                epoch_losses.append(loss_item)
                loss_history.append(loss_item)
                ema_loss = loss_item if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_item
                ema_history.append(ema_loss)

                if len(loss_history) % args.log_every_steps == 0:
                    with open(log_path, "a") as f:
                        f.write(f"Step {len(loss_history)} Loss {loss_item:.6f} EMA {ema_loss:.6f}\n")
                    pbar.set_postfix({"loss": f"{loss_item:.4f}", "EMA": f"{ema_loss:.4f}"})

        # Epoch Cleanup and Checkpointing
        if local_rank == 0:
            epoch_avg = np.mean(epoch_losses)
            ckpt = {
                "adapter": adapter.module.state_dict(),
                "cross_attn": cross_attn.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

            if epoch_avg < best_loss:
                best_loss = epoch_avg
                torch.save(ckpt, os.path.join(args.output_dir, "best_epoch.pt"))

            # Save Loss Curve
            plt.figure(figsize=(6,3))
            plt.plot(loss_history, color='blue', alpha=0.3)
            plt.plot(ema_history, color='red')
            plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
            plt.close()

    for h in handles: h.remove()
    dist.destroy_process_group()

# Argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train Face Landmark Adapter")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--face_latent_dir", type=str, required=True)
    parser.add_argument("--landmark_latent_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a realistic face photo")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--adapter_base_ch", type=int, default=64)
    parser.add_argument("--adapter_target_channels", nargs="+", type=int, default=[320, 640, 1280, 1280])
    parser.add_argument("--adapter_num_heads", nargs="+", type=int, default=[8, 8, 8, 8])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)