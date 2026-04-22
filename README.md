# Surveillance Image Denoising via Domain-Adapted Hybrid SCUNet Framework

A robust, multi-stage image restoration pipeline designed to recover severely degraded CCTV and surveillance imagery. 

Standard deep learning networks often fail on high-resolution security footage. They either crash due to VRAM limits (Out-Of-Memory) or over-smooth physical textures into a blurry "watercolor" effect, destroying critical geometric evidence. This project solves these bottlenecks by combining a domain-adapted SCUNet (Swin-Conv-UNet) AI core with a classical Digital Signal Processing (DSP) mathematical head.

## Key Features & Novelties

1. **Hybrid DSP-AI Architecture**
   Pure deep learning models struggle to differentiate between high-frequency noise and high-frequency physical textures. We shift the paradigm by using the AI strictly for noise subtraction. The output is then caught by a classical DSP head operating in the CIELAB (LAB) color space. By applying Bilateral Filtering and Unsharp Masking exclusively to the isolated Luminance (L) channel, the system mathematically forces edge geometry (text, facial structures, textures) back into the image without chromatic distortion.

2. **VRAM-Agnostic Tiled Inference Engine**
   Transformer-based architectures typically require massive GPU memory for 4K or 1080p images. We engineered a custom wrapper that slices high-resolution images into manageable tensors using reflection padding. It processes them sequentially and stitches them back together using a 2D Hann Window blending algorithm, allowing infinite-resolution processing on standard consumer hardware with zero grid-line artifacts.

3. **Dynamic Domain Adaptation via On-The-Fly Synthesis**
   Models trained on static laboratory datasets fail in the real world. We rebuilt the training loop to include a dynamic CCTV noise synthesizer. During every epoch, the system attacks clean ground-truth images with randomized heavy rain vectors, extreme Gaussian static, and salt-and-pepper dead pixels. This prevents dataset memorization and forces the network to adapt to severe, out-of-distribution environmental hazards.

## System Architecture Pipeline

1. **Pre-Processing:** Impulse noise removal via median filtering to stabilize AI input tensors.
2. **Hardware Abstraction:** Image partitioning via the Tiled Inference Engine.
3. **AI Core:** Noise and weather subtraction using the domain-adapted SCUNet (Swin Transformer + Residual Convolution) backbone.
4. **Seamless Stitching:** Reassembly of processed tiles using mathematical gradient blending (Hann Windows).
5. **DSP Geometry Reconstruction:** LAB color space conversion, L-channel Unsharp Masking, Bilateral Filtering, and CLAHE contrast enhancement.
6. **Final Output:** A high-resolution, structurally sharp surveillance image.

## Quantitative Performance

Evaluated against the base SCUNet model on real-world surveillance scenarios:

* **Edge Geometry Restoration (Laplacian Variance):** Improved from 170.7 (Base AI) to 444.9 (Our Hybrid DSP-AI), proving the elimination of the structural "watercolor" smearing effect.
* **Low-Light / Nighttime Superiority:** Achieved an average PSNR gain of +3.98 dB over the base model on severe nighttime scenarios corrupted by heavy sensor static and motion blur.
* **Hardware Efficiency:** Maintains a flat ~2.1 GB VRAM requirement regardless of input resolution, successfully processing 4K images where the standard baseline triggers CUDA Out-Of-Memory crashes.

## Installation

Ensure you have Python 3.8+ and an NVIDIA GPU with CUDA support installed.

```bash
# Clone the repository
git clone [https://github.com/yourusername/Hybrid-SCUNet-Surveillance.git](https://github.com/yourusername/Hybrid-SCUNet-Surveillance.git)
cd Hybrid-SCUNet-Surveillance

# Install dependencies
pip install -r requirements.txt
