# Synergistic Multi-Modal Fusion for Identity-Preserving Face Aging via Latent Diffusion (MDFA)

Official Implementation of the Paper submitted to The Visual Computer.

**Note:** This code is directly related to the manuscript "Synergistic Multi-Modal Fusion for Identity-Preserving Face Aging via Latent Diffusion" currently under review at The Visual Computer. If you find this work useful, please cite our manuscript.

This project implements MDFA, a structural control adapter for **Stable Diffusion v1.5**.

Traditional face aging often struggles with the trade-off between identity fidelity and realistic geometric transformation. Our framework addresses this by:

**Multi-Level Cross-Attention (MLCA):** Harmonizing dense identity features with sparse geometric priors (landmark heatmaps).

**Latent Adapter (LA):** Hierarchically injecting fused guidance into the frozen U-Net backbone.**

The training dataset from: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq

**Requirement：** 
-- python 3.10 -- pytorch 2.3  -- cuda 12.x 
