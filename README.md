# SAR-KD-Flood

Training code for:

**Distilling Intelligence from Space: Lightweight Deep Learning for SAR-Based Flood Monitoring**

## Overview.

Knowledge Distillation (KD) framework for compressing segmentation models trained on Sentinel-1 Synthetic Aperture Radar (SAR) imagery for flood mapping. Models process single-channel 256x256 patches using a band-agnostic approach — training on all available bands (VV, VH, B2, B3, B4, NDVI, NDWI) while validating on SAR-only (VV, VH).

## Models

| Model | Role | Params | Description |
|-------|------|--------|-------------|
| Unet | Teacher | ~6.6M | U-Net with MobileNetV2 encoder |
| PSPNet | Teacher | ~3.4M | PSPNet with MobileNetV2 encoder |
| MLP_256 | Baseline | ~19K | Per-pixel MLP (128-64-32-16-1) |
| UnetLight | Student | ~12K | Depth-1 U-Net, no skip connections |
| PSPMixer | Student | ~349K | Patch embedding + MLP-Mixer blocks |

## Knowledge Distillation

`kd.py` distills PSPNet (teacher) into PSPMixer (student) using temperature-scaled soft targets:

```
L_total = alpha * L_KD + beta * L_supervised
```

Default: T=4.0, alpha=0.5, beta=0.5. Loss function: Focal Tversky Loss (alpha=0.30, beta=0.70).

## Structure

```
├── lib/
│   ├── model.py          # All 5 architectures
│   ├── loss.py           # Focal Tversky Loss
│   ├── dataset.py        # Band-agnostic dataset
│   └── augmentation.py   # Augmentation pipelines
├── lib_kd/
│   ├── model_kd.py       # Teacher/student creation
│   └── training_kd.py    # KD training loop
├── train_unet.py
├── train_pspnet.py
├── train_mlp256.py
├── train_unetlight.py
├── train_pspmixer.py
└── kd.py                 # PSPNet → PSPMixer distillation
```

## Data

Place data under `02_kfold/`:

```
02_kfold/
├── train/
│   ├── img/   # {patch_id}_{band}.npy
│   └── lab/   # {patch_id}.npy (binary mask)
└── val/
    ├── img/
    └── lab/
```

Normalization: raw values clipped to [-22.225, 0.0] then scaled to [0, 1].

## Usage

Each script is standalone with all parameters embedded at the top:

```bash
python train_unet.py
python train_pspnet.py
python train_mlp256.py
python train_unetlight.py
python train_pspmixer.py
python kd.py
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- segmentation-models-pytorch
- albumentations

## Note

The teacher model in `kd.py` uses ImageNet-pretrained encoder weights via `segmentation-models-pytorch`. For the training dataset (`02_kfold/`) and task-specific pretrained weights, please contact: watcharapong.s@ubu.ac.th

## Acknowledgments and Third-Party Licenses

This project depends on the following open-source libraries:

| Library | License | URL |
|---------|---------|-----|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | https://github.com/pytorch/pytorch |
| [NumPy](https://numpy.org/) | BSD-3-Clause | https://github.com/numpy/numpy |
| [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) | MIT | https://github.com/qubvel-org/segmentation_models.pytorch |
| [albumentations](https://albumentations.ai/) | MIT | https://github.com/albumentations-team/albumentations |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| [OpenCV](https://opencv.org/) | Apache-2.0 | https://github.com/opencv/opencv |

### Pretrained Weights

The MobileNetV2 encoder weights (`encoder_weights='imagenet'`) are loaded at runtime via `timm` and originate from Google's MobileNetV2 release (Apache-2.0). These weights were trained on the [ImageNet](https://image-net.org/) dataset, which is provided for non-commercial research and educational purposes under the [ImageNet Terms of Access](https://image-net.org/download).

> Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.

> Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.

## License

This project is provided for non-commercial research purposes accompanying the publication. Third-party components retain their original licenses as listed above.
