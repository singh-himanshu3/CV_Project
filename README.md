# Surveillance Image Denoising via Domain-Adapted Hybrid SCUNet Framework

[![Open Training Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singh-himanshu3/CV_Project/blob/main/CV_AI_TRAINING.ipynb)
[![Open Inference Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singh-himanshu3/CV_Project/blob/main/CV_Project.ipynb)

---

## Overview

This project presents a domain-adapted image denoising framework for surveillance CCTV enhancement, built upon the Swin-Conv-UNet (SCUNet) architecture. While SCUNet provides strong performance for general-purpose blind denoising, it is not specifically optimized for the complex and heterogeneous noise patterns observed in real-world surveillance imagery.

To address this limitation, we propose a hybrid multi-stage denoising pipeline that combines domain-specific fine-tuning of SCUNet with classical pre-processing and post-processing techniques. A synthetic noise generation strategy is employed to model realistic CCTV degradations, including Gaussian noise, rain streaks, fog, salt-and-pepper noise, luminance corruption, and motion blur artifacts.

The framework achieves average PSNR gains of **+3.93 dB** on daytime scenes and **+8.46 dB** on nighttime scenes over the baseline SCUNet, with consistent SSIM improvements across all test images.

---

## Architecture

![Architecture](assets/architecture.png)

The pipeline follows a three-stage design. The input image is first cleaned of impulse noise via a median filter, then passed through the fine-tuned SCUNet using a memory-efficient tiled inference engine, and finally refined through a hybrid DSP post-processing head operating in the LAB color space. The SCUNet encoder is frozen during training and only the decoder layers (`m_up1`, `m_up2`, `m_up3`, `m_tail`) are fine-tuned on the CCTV-specific noise domain.

---

## Repository Structure

```
CV_Project/
├── CV_AI_TRAINING.ipynb      # Fine-tuning pipeline (run this first)
├── CV_Project.ipynb          # Inference, video processing, and evaluation
└── assets/
    └── architecture.png      # Architecture diagram
```

---

## Getting Started

### Prerequisites

- Google Colab with **T4 GPU** runtime (`Runtime > Change runtime type > T4 GPU`)
- A Google Drive account (for storing datasets and model weights)

### Step 1 — Fine-Tuning (CV_AI_TRAINING.ipynb)

Open the training notebook in Colab and run each cell in order.

**Cell 1** clones the SCUNet repository, installs dependencies, and builds a clean training dataset from skimage standard images and BSD68 patches.

**Cell 2** mounts Google Drive and loads the DIV2K dataset if available. Place your dataset zip at `/content/drive/MyDrive/div2k.zip`.

**Cell 3** defines the `CCTVDataset` class, which synthesizes CCTV degradations on the fly. Each training patch is randomly degraded with a combination of:
- Gaussian noise (sigma 15–55)
- Rain streaks (20–60 random line segments)
- Fog overlay
- Low-light gamma darkening
- JPEG compression (quality 20–60)
- Motion blur (horizontal kernel)

**Cell 4** runs the transfer learning loop. The decoder layers are unfrozen and trained for 5 epochs using a combined L1 + SSIM loss with Adam optimizer (lr = 5e-5). The fine-tuned weights are saved to your Google Drive as `scunet_cctv_weather_finetuned.pth`.

### Step 2 — Inference and Evaluation (CV_Project.ipynb)

Run cells in order after fine-tuning is complete.

| Cell | Description |
|------|-------------|
| Cell 1 | Environment setup, dependency installation, base weight download |
| Cell 2 | Load fine-tuned model and define tiled inference engine |
| Cell 3 | Interactive image enhancement with bilateral filter and sharpening sliders |
| Cell 4 | Full video denoising pipeline — upload an MP4/AVI and download the enhanced output |
| Cell 5 | Live webcam comparison between base SCUNet and fine-tuned model |
| Eval 1.5 | Quantitative PSNR/SSIM evaluation on synthetic CCTV noise with uploaded test images |
| Eval 2.1 | Real CCTV image evaluation with sharpness (Laplacian variance) comparison |

---

## Results

### Comparison with State-of-the-Art (SIDD Benchmark)

| Method | Type | Year | PSNR (dB) |
|--------|------|------|-----------|
| BM3D | Classical | 2007 | 25.65 |
| DnCNN | CNN | 2017 | 32.43 |
| FFDNet | CNN | 2018 | 32.48 |
| RIDNet | CNN | 2019 | 33.24 |
| SwinIR | Transformer | 2021 | 34.52 |
| SCUNet (Base) | Hybrid | 2023 | 34.78 |
| **Ours (Hybrid DSP-AI)** | **DSP-Hybrid** | **2026** | **35.12** |

### Daytime Scene Results

| Scene | Noisy PSNR | Base PSNR | Ours PSNR | Gain |
|-------|-----------|-----------|-----------|------|
| Multi-lane Highway | 17.56 | 18.67 | 21.59 | +2.92 |
| City Aerial Grid | 15.69 | 16.09 | 20.30 | +4.21 |
| Crowded Street | 16.75 | 17.72 | 22.39 | +4.66 |
| **Average** | | | | **+3.93 dB** |

### Nighttime Scene Results

| Scene | Noisy PSNR | Base PSNR | Ours PSNR | Gain |
|-------|-----------|-----------|-----------|------|
| City Skyline | 15.11 | 16.46 | 24.23 | +7.77 |
| Night Highway | 15.15 | 15.69 | 23.24 | +7.56 |
| Street Road | 14.48 | 14.89 | 24.93 | +10.05 |
| **Average** | | | | **+8.46 dB** |

The proposed method is particularly effective under low-light conditions, where luminance noise and compression artifacts are the dominant degradation factors.

### Combined PSNR and SSIM Summary

| Scene | Noisy PSNR | Base PSNR | Ours PSNR | Noisy SSIM | Base SSIM | Ours SSIM |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Multi-lane Highway | 17.56 | 18.67 | 21.59 | 0.453 | 0.586 | 0.630 |
| City Aerial Grid | 15.69 | 16.09 | 20.30 | 0.522 | 0.649 | 0.673 |
| Crowded Street | 16.75 | 17.72 | 22.39 | 0.521 | 0.732 | 0.779 |
| City Skyline | 15.11 | 16.46 | 24.23 | 0.269 | 0.606 | 0.742 |
| Night Highway | 15.15 | 15.69 | 23.24 | 0.247 | 0.621 | 0.705 |
| Street Road | 14.48 | 14.89 | 24.93 | 0.278 | 0.539 | 0.624 |

---

## Key Contributions

1. A CCTV-specific noise synthesis pipeline that captures key characteristics of surveillance degradation including impulse noise, motion streaks, and luminance corruption.
2. A domain adaptation strategy for SCUNet through selective fine-tuning of decoder (upsampling) layers on the synthesized dataset.
3. A hybrid DSP-AI post-processing head operating in the LAB color space to mathematically restore edge geometry and prevent AI-induced structural smearing.
4. A memory-efficient tiled inference engine using Hanning window blending, enabling processing of 4K footage without GPU memory failures.
5. Quantitative improvements in both PSNR and SSIM over the baseline across daytime and nighttime surveillance scenarios.
6. Practical end-to-end system design validated on real CCTV images, uploaded videos, and live webcam feeds.

---

## Dependencies

All dependencies are installed automatically inside the notebooks. The key libraries are:

```
torch
timm
einops
thop
scikit-image
opencv-python
pytorch-msssim
tqdm
matplotlib
ipywidgets
```

The notebooks use the SCUNet architecture from [cszn/SCUNet](https://github.com/cszn/SCUNet).

---

## Model Weights

The fine-tuned model weights can be downloaded from the following Google Drive link:

[Download Fine-tuned Weights](https://drive.google.com/file/d/12J2rdPImvxzhcLVL65QrALe7AfHlMBdG/view?usp=sharing)

Place the downloaded `.pth` file in your Google Drive root before running the inference notebook.

---

## Acknowledgements

This project builds on the SCUNet architecture proposed by Zhang et al. (2023). We thank the authors of SCUNet, SwinIR, and Restormer for releasing their code and pre-trained models publicly.

---

## License

This project is released for academic and research use only.
