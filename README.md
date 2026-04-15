# AI-Powered Forensic Image & Video Enhancement System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)

This repository contains a hybrid Digital Signal Processing (DSP) and Artificial Intelligence (AI) architecture designed for the restoration of severely degraded surveillance footage. The system is engineered to resolve the limitations of standard deep learning models, specifically hardware memory constraints and structural hallucination in high-noise environments.

## System Architecture

The pipeline integrates a **Swin-Transformer UNet (SCUNet)** with a classical DSP post-processing head. 

### Key Technical Contributions
1. **VRAM-Agnostic Tiled Inference:** A custom sliding-window algorithm utilizing Reflection Padding and Hann-Window blending. This allows high-resolution frames (e.g., 4K) to be processed on constrained consumer GPUs without grid artifacts.
2. **Hybrid DSP-AI Head:** A multi-stage processing pipeline consisting of 3x3 Median Pre-processing, SCUNet AI inference, Non-Local Means (NLM) denoising, and LAB-Space CLAHE sharpening.
3. **Dynamic Data Synthesis:** Implements on-the-fly mathematical injection of Gaussian noise and simulated rain vectors into the DIV2K dataset during the training loop to prevent model overfitting.
4. **Live Deployment Bridge:** Features a JavaScript-to-Python bridge enabling near real-time webcam frame capture and forensic analysis.

## Pre-trained Weights

Due to repository file size constraints, the custom-trained `.pth` SCUNet weights are hosted externally. 

**[Download Pre-trained SCUNet Weights]( https://drive.google.com/file/d/1Z9e4bhXhA74gvK-cRvQKXavDGFOO3mAD/view?usp=sharing)**

*Note: The weights file must be downloaded and mounted to your active environment prior to executing the inference scripts.*

## Repository Structure

* `CV_Project.ipynb` - The primary deployment notebook. Contains the tiled inference engine, the JavaScript webcam bridge, and the temporal video processing pipeline.
* `FORENSIC_AI_TRAINING.ipynb` - The transfer learning and data synthesis script used to train the model.
* `CV_Report_Full_and_final.pdf` - Comprehensive technical documentation detailing system architecture, loss metrics, and performance evaluation.

## Execution Guide

The system is optimized for cloud execution via Google Colaboratory.

1. Open `CV_Project.ipynb` in Google Colab.
2. Ensure the hardware accelerator is set to **T4 GPU** (`Runtime` > `Change runtime type`).
3. Upload the `.pth` weights file to your Google Drive and verify the path mapping in the environment setup cell.
4. Execute the cells sequentially to initialize the engine.
5. Provide input via the JavaScript webcam tool or by uploading target `.mp4` or `.png` files for enhancement.

## Performance Evaluation

The hybrid pipeline has been validated against severe degradation scenarios, including heavy Gaussian noise, synthetic rain, and Salt & Pepper sensor artifacts. The system successfully:
* Eliminates mathematical static and weather interference.
* Recovers underlying structural geometry without introducing hallucinated artifacts.
* Suppresses deep-learning-induced "watercolor" smearing via NLM processing.
* Preserves natural scene lighting utilizing isolated L-channel (LAB space) unsharp masking.

For a detailed quantitative breakdown, including accuracy metrics and confusion matrices, refer to the included PDF report.
