# DeVeiler

This repository contains the minimal training code for the three-stage DeVeiler pipeline.

## Overview

DeVeiler is trained in three stages:

| Stage | Config | Model | Purpose |
|---|---|---|---|
| 1 | `options/train/DeVeiler/stage1_pretrain.yml` | `deveiler_wo_vgcm` | Train the base restoration model. |
| 2 | `options/train/DeVeiler/stage2_reblur.yml` | `DDN` | Train the degradation / reblur network. |
| 3 | `options/train/DeVeiler/stage3_finetune.yml` | `deveiler` + `DDN_frozen` | Load Stage 1 and Stage 2 checkpoints, then finetune the final DeVeiler model. |

## Environment

Create a Python environment and install PyTorch first. Choose the PyTorch command that matches your CUDA version from the official PyTorch website.

Example:

```bash
conda create -n deveiler python=3.9 -y
conda activate deveiler

# Install PyTorch according to your CUDA version.
# Example only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .
```

If `lpips` is not installed by your environment, install it manually:

```bash
pip install lpips
```

## Configure Data Paths

Before training, edit the three config files under:

```text
options/train/DeVeiler/
```

The public configs use placeholder paths such as:

```text
/path/to/src_gt
/path/to/src_lq
/path/to/clear_images
/path/to/veilgen_generate_images
/path/to/individual_lpe_maps_folder
/path/to/checkpoints/stage1_pretrain_net_g.pth
/path/to/checkpoints/stage2_reblur_net_g.pth
```

Replace them with your local paths.

Expected paired image folders:

```text
src_gt/
  image_000.png
  image_001.png

src_lq/
  image_000.png
  image_001.png
```

For Stage 2 and Stage 3, LPE maps are loaded by `MixedPairedImageDataset_with_LPE_Map`. Keep the image names aligned with the corresponding LPE map files expected by your dataset implementation.

## Stage 1: DeVeiler Pretraining

Edit:

```text
options/train/DeVeiler/stage1_pretrain.yml
```

Important fields:

```yaml
datasets:
  train:
    dataroot_gt: /path/to/src_gt
    dataroot_lq: /path/to/src_lq
  val_1:
    dataroot_gt: /path/to/screen_tgt_gt_all
    dataroot_lq: /path/to/screen_tgt_lq
```

Run:

```bash
bash scripts/train_deveiler_stage1.sh
```

Equivalent command:

```bash
python basicsr/train.py -opt options/train/DeVeiler/stage1_pretrain.yml
```

After training, use the Stage 1 checkpoint as `path.pretrain_network_g` in Stage 3.

## Stage 2: DDN Training

Edit:

```text
options/train/DeVeiler/stage2_reblur.yml
```

Important fields:

```yaml
datasets:
  train:
    dataroot_lq:
      - /path/to/src_gt
      - /path/to/clear_images
    dataroot_gt:
      - /path/to/src_lq
      - /path/to/veilgen_generate_images
    LPE_map_folder:
      - null
      - /path/to/lpe_maps/generated_reblur_train
```

Run:

```bash
bash scripts/train_deveiler_stage2_reblur.sh
```

Equivalent command:

```bash
python basicsr/train.py -opt options/train/DeVeiler/stage2_reblur.yml
```

After training, use the Stage 2 checkpoint as `path.pretrain_network_r` in Stage 3.

## Stage 3: DeVeiler Finetuning

Edit:

```text
options/train/DeVeiler/stage3_finetune.yml
```

Set the Stage 1 and Stage 2 checkpoints:

```yaml
path:
  pretrain_network_g: /path/to/checkpoints/stage1_pretrain_net_g.pth
  pretrain_network_r: /path/to/checkpoints/stage2_reblur_net_g.pth
```

Run:

```bash
bash scripts/train_deveiler_stage3_finetune.sh
```

Equivalent command:

```bash
python basicsr/train.py -opt options/train/DeVeiler/stage3_finetune.yml
```


