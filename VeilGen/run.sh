#step1
CUDA_VISIBLE_DEVICES=0 accelerate launch train_veilgen.py --config configs/train/train_veilgen.yaml

#step2
python predict_latent_maps.py --config configs/predict_latent_maps.yaml --image_folder /path/to/real/images --output_dir ./outputs/latent_maps --individual_save_dir ./outputs/latent_maps/individual --stacked_save_dir ./outputs/latent_maps/stacked --device cuda:0 --timesteps 0

#step3
CUDA_VISIBLE_DEVICES=0 python infer_veilgen.py --config configs/inference/infer_veilgen.yaml
