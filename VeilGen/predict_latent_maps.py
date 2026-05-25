import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from diffbir.model import ControlLDM, Diffusion
from diffbir.sampler import SpacedSampler

# LOTGMP is the latent-map predictor used by VeilGen.
# scale_vector represents the latent transmission map.
# shift_vector represents the latent glare map.

def get_cfg_value(node, name, fallback_name=None, default=None):
    if hasattr(node, name):
        return getattr(node, name)
    if fallback_name and hasattr(node, fallback_name):
        return getattr(node, fallback_name)
    return default

def get_lotgmp_config(model_cfg):
    if hasattr(model_cfg, 'lotgmp'):
        return model_cfg.lotgmp
    return model_cfg.lpe

class ControlLDMWithLOTGMP(ControlLDM):

    def __init__(self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor, lotgmp_cfg):
        super().__init__(unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor)
        self.lotgmp = instantiate_from_config(lotgmp_cfg)
        self._check_and_modify_controlnet_first_layer()

    def _check_and_modify_controlnet_first_layer(self):
        first_conv = self.controlnet.input_blocks[0][0]
        current_channels = first_conv.in_channels
        expected_channels = 8
        if current_channels == expected_channels:
            return
        new_conv = torch.nn.Conv2d(expected_channels, first_conv.out_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias is not None, first_conv.padding_mode)
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if first_conv.bias is not None:
                torch.nn.init.zeros_(new_conv.bias)
        self.controlnet.input_blocks[0][0] = new_conv

class MemorySafeLOTGMPAnalyzer:

    def __init__(self, config_path, device='cuda:0'):
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(config_path)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        self.scale_stats = []
        self.shift_stats = []
        self.scale_tensors = []
        self.shift_tensors = []
        print(f'Using device: {self.device}')

    def _create_models(self):
        print('Creating models...')
        self.cldm = ControlLDMWithLOTGMP(unet_cfg=self.cfg.model.cldm.params.unet_cfg, vae_cfg=self.cfg.model.cldm.params.vae_cfg, clip_cfg=self.cfg.model.cldm.params.clip_cfg, controlnet_cfg=self.cfg.model.cldm.params.controlnet_cfg, latent_scale_factor=self.cfg.model.cldm.params.latent_scale_factor, lotgmp_cfg=get_lotgmp_config(self.cfg.model))
        self.diffusion = instantiate_from_config(self.cfg.model.diffusion)
        print('Models created successfully!')

    def _load_weights(self):
        print('Loading weights...')
        sd = torch.load(self.cfg.inference.sd_path, map_location='cpu')['state_dict']
        unused, missing = self.cldm.load_pretrained_sd(sd)
        print(f'Loaded SD weights. Unused: {len(unused)}, Missing: {len(missing)}')
        if hasattr(self.cfg.inference, 'controlnet_path') and self.cfg.inference.controlnet_path:
            self.cldm.load_controlnet_from_ckpt(torch.load(self.cfg.inference.controlnet_path, map_location='cpu'))
            print(f'Loaded ControlNet weights from: {self.cfg.inference.controlnet_path}')
        lotgmp_path = get_cfg_value(self.cfg.inference, 'lotgmp_path', 'lpe_path')
        if lotgmp_path:
            lotgmp_state_dict = torch.load(lotgmp_path, map_location='cpu')
            self.cldm.lotgmp.load_state_dict(lotgmp_state_dict)
            print(f'Loaded LOTGMP weights from: {lotgmp_path}')
        else:
            print('Warning: No LOTGMP weights provided!')
        self.cldm.eval().to(self.device)
        self.diffusion.to(self.device)
        print('All weights loaded successfully!')

    def _clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()

    def encode_image(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            if image_tensor.max() <= 1.0:
                image_tensor = image_tensor * 2 - 1
            z = self.cldm.vae_encode(image_tensor)
            del image_tensor
            self._clear_gpu_memory()
            return z

    def analyze_single_image(self, image_path, timesteps=None, save_dir=None):
        if timesteps is None:
            timesteps = [111, 222, 333, 444, 555, 666, 777, 888, 999]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)
        _, _, h, w = image.shape
        target_h, target_w = (h // 2, w // 2)
        rescaler = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = rescaler(image)
        image = pad_to_multiples_of(image, multiple=64)
        z_blur = self.encode_image(image)
        del image
        self._clear_gpu_memory()
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        results = {'image_path': image_path, 'z_blur_shape': z_blur.shape, 'timesteps': timesteps, 'predictions': []}
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            noise = torch.randn_like(z_blur).to(self.device)
            x_noisy = self.diffusion.q_sample(z_blur, t_tensor, noise)
            del noise
            self._clear_gpu_memory()
            with torch.no_grad():
                lotgmp_output = self.cldm.lotgmp(x_noisy, z_blur, t_tensor)
                scale_vector = lotgmp_output['scale_vector']
                shift_vector = lotgmp_output['shift_vector']
            scale_vector_cpu = scale_vector.cpu()
            shift_vector_cpu = shift_vector.cpu()
            del scale_vector, shift_vector, x_noisy, lotgmp_output
            self._clear_gpu_memory()
            scale_numpy = scale_vector_cpu.numpy()
            shift_numpy = shift_vector_cpu.numpy()
            scale_numpy = scale_numpy[:, :, :target_h, :target_w]
            shift_numpy = shift_numpy[:, :, :target_h, :target_w]
            if save_dir is not None:
                if len(timesteps) > 1:
                    scale_filename = f'{image_basename}_scale_t{t:03d}.npy'
                    shift_filename = f'{image_basename}_shift_t{t:03d}.npy'
                else:
                    scale_filename = f'{image_basename}_scale.npy'
                    shift_filename = f'{image_basename}_shift.npy'
                scale_filepath = os.path.join(save_dir, scale_filename)
                shift_filepath = os.path.join(save_dir, shift_filename)
                np.save(scale_filepath, scale_numpy)
                np.save(shift_filepath, shift_numpy)
            self.scale_tensors.append(scale_numpy)
            self.shift_tensors.append(shift_numpy)
            scale_stats = {'mean': scale_vector_cpu.mean().item(), 'std': scale_vector_cpu.std().item(), 'min': scale_vector_cpu.min().item(), 'max': scale_vector_cpu.max().item(), 'abs_mean': scale_vector_cpu.abs().mean().item()}
            shift_stats = {'mean': shift_vector_cpu.mean().item(), 'std': shift_vector_cpu.std().item(), 'min': shift_vector_cpu.min().item(), 'max': shift_vector_cpu.max().item(), 'abs_mean': shift_vector_cpu.abs().mean().item()}
            results['predictions'].append({'timestep': t, 'scale_stats': scale_stats, 'shift_stats': shift_stats})
            self.scale_stats.append(scale_stats)
            self.shift_stats.append(shift_stats)
            del scale_vector_cpu, shift_vector_cpu
        del z_blur
        self._clear_gpu_memory()
        return results

    def analyze_dataset(self, image_folder, timesteps=None, max_images=None, save_dir=None):
        print(f'Analyzing dataset in: {image_folder}')
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            print(f'Individual results will be saved to: {save_dir}')
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        if max_images:
            image_paths = image_paths[:max_images]
        print(f'Found {len(image_paths)} images to analyze')
        all_results = []
        for i, image_path in enumerate(tqdm(image_paths, desc='Analyzing images')):
            try:
                self._clear_gpu_memory()
                result = self.analyze_single_image(image_path, timesteps, save_dir)
                all_results.append(result)
            except Exception as e:
                print(f'Error processing {image_path}: {e}')
                self._clear_gpu_memory()
                continue
        return all_results

    def compute_global_statistics(self):
        if not self.scale_stats:
            return None
        scale_means = [s['mean'] for s in self.scale_stats]
        scale_stds = [s['std'] for s in self.scale_stats]
        scale_abs_means = [s['abs_mean'] for s in self.scale_stats]
        shift_means = [s['mean'] for s in self.shift_stats]
        shift_stds = [s['std'] for s in self.shift_stats]
        shift_abs_means = [s['abs_mean'] for s in self.shift_stats]
        global_stats = {'num_predictions': len(self.scale_stats), 'scale': {'per_prediction_mean': np.mean(scale_means), 'per_prediction_std': np.mean(scale_stds), 'per_prediction_abs_mean': np.mean(scale_abs_means), 'mean_of_means': np.mean(scale_means), 'std_of_means': np.std(scale_means), 'min_of_mins': np.min([s['min'] for s in self.scale_stats]), 'max_of_maxs': np.max([s['max'] for s in self.scale_stats])}, 'shift': {'per_prediction_mean': np.mean(shift_means), 'per_prediction_std': np.mean(shift_stds), 'per_prediction_abs_mean': np.mean(shift_abs_means), 'mean_of_means': np.mean(shift_means), 'std_of_means': np.std(shift_means), 'min_of_mins': np.min([s['min'] for s in self.shift_stats]), 'max_of_maxs': np.max([s['max'] for s in self.shift_stats])}}
        return global_stats

    def plot_distributions(self, save_path=None):
        if not self.scale_stats:
            print('No data to plot!')
            return
        scale_means = [s['mean'] for s in self.scale_stats]
        scale_stds = [s['std'] for s in self.scale_stats]
        shift_means = [s['mean'] for s in self.shift_stats]
        shift_stds = [s['std'] for s in self.shift_stats]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LOTGMP Prediction Distributions', fontsize=16)
        axes[0, 0].hist(scale_means, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Scale Mean Distribution')
        axes[0, 0].set_xlabel('Scale Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].hist(scale_stds, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Scale Std Distribution')
        axes[0, 1].set_xlabel('Scale Std')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].hist(shift_means, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Shift Mean Distribution')
        axes[1, 0].set_xlabel('Shift Mean')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].hist(shift_stds, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Shift Std Distribution')
        axes[1, 1].set_xlabel('Shift Std')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Distribution plot saved to: {save_path}')
        else:
            plt.show()
        plt.close()

    def save_statistics(self, save_path):
        global_stats = self.compute_global_statistics()
        if global_stats is None:
            print('No statistics to save!')
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, scale_stats=self.scale_stats, shift_stats=self.shift_stats, global_stats=global_stats)
        print(f'Statistics saved to: {save_path}')
        base_path = save_path.replace('.npz', '')
        scale_tensors_path = f'{base_path}_scale_tensors.npy'
        shift_tensors_path = f'{base_path}_shift_tensors.npy'
        if self.scale_tensors:
            scale_array = np.stack(self.scale_tensors, axis=0)
            np.save(scale_tensors_path, scale_array)
            print(f'Scale tensors saved to: {scale_tensors_path}')
        if self.shift_tensors:
            shift_array = np.stack(self.shift_tensors, axis=0)
            np.save(shift_tensors_path, shift_array)
            print(f'Shift tensors saved to: {shift_tensors_path}')

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to input images')
    parser.add_argument('--output_dir', type=str, default='./lotgmp_analysis_results', help='Output directory')
    parser.add_argument('--individual_save_dir', type=str, default=None, help='Directory for per-image scale and shift outputs')
    parser.add_argument('--stacked_save_dir', type=str, default=None, help='Directory for stacked scale and shift outputs')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--timesteps', type=int, nargs='+', default=[111], help='Timesteps to analyze')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    individual_save_dir = args.individual_save_dir or os.path.join(args.output_dir, 'individual')
    stacked_save_dir = args.stacked_save_dir or os.path.join(args.output_dir, 'stacked')
    os.makedirs(individual_save_dir, exist_ok=True)
    os.makedirs(stacked_save_dir, exist_ok=True)
    analyzer = MemorySafeLOTGMPAnalyzer(args.config, args.device)
    analyzer._create_models()
    analyzer._load_weights()
    results = analyzer.analyze_dataset(image_folder=args.image_folder, timesteps=args.timesteps, max_images=args.max_images, save_dir=individual_save_dir)
    stats_path = os.path.join(stacked_save_dir, 'lotgmp_statistics.npz')
    analyzer.save_statistics(stats_path)
    plot_path = os.path.join(args.output_dir, 'lotgmp_distributions.png')
    analyzer.plot_distributions(plot_path)
    print(f'\nAnalysis completed! Results saved to: {args.output_dir}')
    print(f'Individual image results saved to: {individual_save_dir}')
    print(f'Stacked tensor results saved to: {stacked_save_dir}')
if __name__ == '__main__':
    main()
