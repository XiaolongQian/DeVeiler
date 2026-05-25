# VeilGen

VeilGen is a latent-map-guided image restoration framework. The LOTGMP module predicts latent maps used by the ControlNet branch:

- `scale_vector`: latent transmission map
- `shift_vector`: latent glare map

This folder contains the minimal training, latent-map prediction, and inference code prepared for release.

## Installation

Create a clean Python environment and install the dependencies:

```bash
conda create -n veilgen python=3.10 -y
conda activate veilgen
pip install -r requirements.txt
```

The provided `requirements.txt` uses PyTorch with CUDA 11.8. If your CUDA version is different, install the matching PyTorch build first, then install the remaining packages.

## Required Paths

Before running VeilGen, update the placeholder paths in the config files.

### Training Config

Edit `configs/train/train_veilgen.yaml`:

- `dataset.train.params.lq_folder`: paired low-quality training images
- `dataset.train.params.gt_folder`: paired ground-truth training images
- `dataset.train.params.real_folder`: real degraded training images
- `dataset.train.params.paired_meta_file`: optional paired image list
- `dataset.train.params.real_meta_file`: optional real image list
- `train.sd_path`: pretrained Stable Diffusion checkpoint
- `train.exp_dir`: training output directory
- `train.resume`: optional ControlNet checkpoint for resume

### Latent-Map Prediction Config

Edit `configs/predict_latent_maps.yaml`:

- `inference.sd_path`: pretrained Stable Diffusion checkpoint
- `inference.controlnet_path`: trained ControlNet checkpoint
- `inference.lotgmp_path`: trained LOTGMP checkpoint

The command-line `--image_folder` should point to the real images used to estimate latent maps.

### Inference Config

Edit `configs/inference/infer_veilgen.yaml`:

- `inference.sd_path`: pretrained Stable Diffusion checkpoint
- `inference.controlnet_path`: trained ControlNet checkpoint
- `inference.lotgmp_path`: trained LOTGMP checkpoint
- `inference.lotgmp_modulation_path`: trained LOTGMP modulation checkpoint
- `inference.image_folder`: input images for inference
- `inference.result_folder`: output directory
- `inference.lotgmp_scale_path`: predicted scale tensor file
- `inference.lotgmp_shift_path`: predicted shift tensor file

## Usage

Run all commands from the `VeilGen/` directory.

### Train

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_veilgen.py --config configs/train/train_veilgen.yaml
```

Training checkpoints are saved under `train.exp_dir/checkpoints`.

### Predict Latent Maps

```bash
CUDA_VISIBLE_DEVICES=0 python predict_latent_maps.py --config configs/predict_latent_maps.yaml --image_folder /path/to/real/images --output_dir ./outputs/latent_maps --individual_save_dir ./outputs/latent_maps/individual --stacked_save_dir ./outputs/latent_maps/stacked --device cuda:0 --timesteps 0
```

This produces files such as:

- `./outputs/latent_maps/individual/*_scale.npy`
- `./outputs/latent_maps/individual/*_shift.npy`
- `./outputs/latent_maps/stacked/lotgmp_statistics_scale_tensors.npy`
- `./outputs/latent_maps/stacked/lotgmp_statistics_shift_tensors.npy`
- `./outputs/latent_maps/stacked/lotgmp_statistics.npz`
- `./outputs/latent_maps/lotgmp_distributions.png`

Use the stacked scale and shift tensor paths in `configs/inference/infer_veilgen.yaml`.

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer_veilgen.py --config configs/inference/infer_veilgen.yaml
```

Results are written to `inference.result_folder`.
