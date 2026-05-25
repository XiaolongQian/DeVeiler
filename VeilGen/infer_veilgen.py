import os
import cv2
import glob
import copy
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from diffbir.sampler import SpacedSampler
from diffbir.model import ControlLDM, Diffusion
from diffbir.pipeline import pad_to_multiples_of
from diffbir.utils.common import instantiate_from_config
from torchvision.transforms import ToTensor, Resize, InterpolationMode
from models.lotgmp import create_lotgmp
from diffbir.model.unet import ResBlock, TimestepEmbedSequential, TimestepBlock
from diffbir.model.attention import SpatialTransformer
import torch.nn as nn

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

class ControlNetWithLOTGMP(nn.Module):

    def __init__(self, base_controlnet, lotgmp_cfg):
        super().__init__()
        self.base_controlnet = base_controlnet
        self.lotgmp_modulation_layers = nn.ModuleList()
        self._add_lotgmp_modulation_layers()

    def _add_lotgmp_modulation_layers(self):
        for module in self.base_controlnet.input_blocks:
            if hasattr(module, '__iter__'):
                for submodule in module:
                    if isinstance(submodule, ResBlock):
                        channels = submodule.out_channels
                        lotgmp_layer = nn.ModuleDict({'scale_conv': nn.Conv2d(4, channels, 1, 1, 0), 'shift_conv': nn.Conv2d(4, channels, 1, 1, 0)})
                        self.lotgmp_modulation_layers.append(lotgmp_layer)
        for submodule in self.base_controlnet.middle_block:
            if isinstance(submodule, ResBlock):
                channels = submodule.out_channels
                lotgmp_layer = nn.ModuleDict({'scale_conv': nn.Conv2d(4, channels, 1, 1, 0), 'shift_conv': nn.Conv2d(4, channels, 1, 1, 0)})
                self.lotgmp_modulation_layers.append(lotgmp_layer)

    def load_lotgmp_modulation_weights(self, lotgmp_modulation_ckpt_path):
        lotgmp_modulation_state_dict = torch.load(lotgmp_modulation_ckpt_path, map_location='cpu')
        self.lotgmp_modulation_layers.load_state_dict(lotgmp_modulation_state_dict)
        print(f'Loaded LOTGMP modulation weights from: {lotgmp_modulation_ckpt_path}')

    def forward(self, x, hint, timesteps, context, lotgmp_params=None):
        if lotgmp_params is None:
            batch_size = x.shape[0]
            lotgmp_params = {'scale_vector': torch.zeros_like(x), 'shift_vector': torch.zeros_like(x), 'uncond_mask': torch.zeros(batch_size, dtype=torch.bool, device=x.device)}
        return self._forward_with_lotgmp(x, hint, timesteps, context, lotgmp_params)

    def _forward_with_lotgmp(self, x, hint, timesteps, context, lotgmp_params):
        scale_vector = lotgmp_params['scale_vector']
        shift_vector = lotgmp_params['shift_vector']
        from diffbir.model.util import timestep_embedding
        t_emb = timestep_embedding(timesteps, self.base_controlnet.model_channels, repeat_only=False)
        emb = self.base_controlnet.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []
        h, emb, context = map(lambda t: t.type(self.base_controlnet.dtype), (x, emb, context))
        lotgmp_layer_idx = 0
        for module, zero_conv in zip(self.base_controlnet.input_blocks, self.base_controlnet.zero_convs):
            enhanced_module = self._create_enhanced_module_sequence(module, lotgmp_params, lotgmp_layer_idx)
            h = enhanced_module(h, emb, context)
            lotgmp_layer_idx = self._count_resblocks_in_module(module, lotgmp_layer_idx)
            outs.append(zero_conv(h, emb, context))
        enhanced_middle_block = self._create_enhanced_module_sequence(self.base_controlnet.middle_block, lotgmp_params, lotgmp_layer_idx)
        h = enhanced_middle_block(h, emb, context)
        outs.append(self.base_controlnet.middle_block_out(h, emb, context))
        return outs

    def _create_enhanced_module_sequence(self, module, lotgmp_params, start_lotgmp_idx):
        enhanced_layers = []
        lotgmp_idx = start_lotgmp_idx
        if hasattr(module, '__iter__'):
            for submodule in module:
                if isinstance(submodule, ResBlock):
                    enhanced_resblock = self._create_enhanced_resblock(submodule, lotgmp_params, lotgmp_idx)
                    enhanced_layers.append(enhanced_resblock)
                    lotgmp_idx += 1
                else:
                    enhanced_layers.append(submodule)
        else:
            enhanced_layers.append(module)
        return TimestepEmbedSequential(*enhanced_layers)

    def _create_enhanced_resblock(self, resblock, lotgmp_params, lotgmp_idx):

        class EnhancedResBlock(TimestepBlock):

            def __init__(self, base_resblock, lotgmp_modulation_layer, lotgmp_params, lotgmp_idx):
                super().__init__()
                self.base_resblock = base_resblock
                self.lotgmp_modulation_layer = lotgmp_modulation_layer
                self.lotgmp_params = lotgmp_params
                self.lotgmp_idx = lotgmp_idx

            def forward(self, x, emb):
                h_processed = self._forward_resblock_without_output_modulation(x, emb, self.base_resblock)
                if self.lotgmp_modulation_layer is not None:
                    scale_vector = self.lotgmp_params['scale_vector']
                    shift_vector = self.lotgmp_params['shift_vector']
                    B, C, H, W = h_processed.shape
                    scale_resized = torch.nn.functional.interpolate(scale_vector, size=(H, W), mode='bilinear', align_corners=False)
                    shift_resized = torch.nn.functional.interpolate(shift_vector, size=(H, W), mode='bilinear', align_corners=False)
                    scale_proj = self.lotgmp_modulation_layer['scale_conv'](scale_resized)
                    shift_proj = self.lotgmp_modulation_layer['shift_conv'](shift_resized)
                    h_final = h_processed * (1 + scale_proj) + shift_proj
                else:
                    h_final = h_processed
                return self.base_resblock.skip_connection(x) + h_final

            def _forward_resblock_without_output_modulation(self, h, emb, resblock):
                if resblock.updown:
                    in_rest, in_conv = (resblock.in_layers[:-1], resblock.in_layers[-1])
                    h = in_rest(h)
                    h = resblock.h_upd(h)
                    x = resblock.x_upd(h)
                    h = in_conv(h)
                else:
                    h = resblock.in_layers(h)
                emb_out = resblock.emb_layers(emb).type(h.dtype)
                while len(emb_out.shape) < len(h.shape):
                    emb_out = emb_out[..., None]
                if resblock.use_scale_shift_norm:
                    out_norm, out_rest = (resblock.out_layers[0], resblock.out_layers[1:])
                    scale, shift = torch.chunk(emb_out, 2, dim=1)
                    h = out_norm(h) * (1 + scale) + shift
                    h = out_rest(h)
                else:
                    h = h + emb_out
                    h = resblock.out_layers(h)
                return h
        lotgmp_modulation_layer = None
        if lotgmp_idx < len(self.lotgmp_modulation_layers):
            lotgmp_modulation_layer = self.lotgmp_modulation_layers[lotgmp_idx]
        return EnhancedResBlock(resblock, lotgmp_modulation_layer, lotgmp_params, lotgmp_idx)

    def _count_resblocks_in_module(self, module, start_idx):
        count = start_idx
        if hasattr(module, '__iter__'):
            for submodule in module:
                if isinstance(submodule, ResBlock):
                    count += 1
        return count

class ControlLDMWithLOTGMPEnhanced(ControlLDM):

    def __init__(self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor, lotgmp_cfg):
        super().__init__(unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor)
        self.lotgmp = instantiate_from_config(lotgmp_cfg)
        self.controlnet_enhanced = ControlNetWithLOTGMP(self.controlnet, lotgmp_cfg)

    def load_lotgmp_weights(self, lotgmp_ckpt_path):
        lotgmp_state_dict = torch.load(lotgmp_ckpt_path, map_location='cpu')
        self.lotgmp.load_state_dict(lotgmp_state_dict)
        print(f'Loaded LOTGMP weights from: {lotgmp_ckpt_path}')

    def load_lotgmp_modulation_weights(self, lotgmp_modulation_ckpt_path):
        self.controlnet_enhanced.load_lotgmp_modulation_weights(lotgmp_modulation_ckpt_path)

    def set_fixed_lotgmp_params(self, x_noisy, device, scale_path=None, shift_path=None, use_average=False):
        if scale_path is None or shift_path is None:
            raise ValueError('scale_path and shift_path must be provided')
        scale_vector = np.load(scale_path)
        shift_vector = np.load(shift_path)
        scale_vector = torch.from_numpy(scale_vector).to(device)
        shift_vector = torch.from_numpy(shift_vector).to(device)
        max_idx = scale_vector.shape[0]
        if use_average:
            scale_vector = scale_vector.mean(dim=0, keepdim=True).squeeze(0)
            shift_vector = shift_vector.mean(dim=0, keepdim=True).squeeze(0)
        else:
            np.random.seed(42)
            idx = np.random.randint(0, max_idx)
            scale_vector = scale_vector[idx]
            shift_vector = shift_vector[idx]
        self._fixed_scale_vector = scale_vector
        self._fixed_shift_vector = shift_vector

    def forward(self, x_noisy, t, cond):
        c_txt = cond['c_txt']
        c_img = cond['c_img']
        batch_size = x_noisy.shape[0]
        uncond_mask = torch.all(c_img == 0, dim=(1, 2, 3))
        has_uncond_samples = torch.any(uncond_mask)
        if has_uncond_samples:
            if hasattr(self, '_fixed_scale_vector'):
                scale_vector = self._fixed_scale_vector.repeat(batch_size, 1, 1, 1)
                shift_vector = self._fixed_shift_vector.repeat(batch_size, 1, 1, 1)
                scale_vector = torch.where(uncond_mask.view(-1, 1, 1, 1), scale_vector, torch.zeros_like(scale_vector))
                shift_vector = torch.where(uncond_mask.view(-1, 1, 1, 1), shift_vector, torch.zeros_like(shift_vector))
            else:
                scale_vector = torch.zeros_like(x_noisy)
                shift_vector = torch.zeros_like(x_noisy)
        else:
            scale_vector = torch.zeros_like(x_noisy)
            shift_vector = torch.zeros_like(x_noisy)
        lotgmp_params = {'scale_vector': scale_vector, 'shift_vector': shift_vector, 'uncond_mask': uncond_mask}
        control = self.controlnet_enhanced(x=x_noisy, hint=c_img, timesteps=t, context=c_txt, lotgmp_params=lotgmp_params)
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(x=x_noisy, timesteps=t, context=c_txt, control=control, only_mid_control=False)
        return eps

def sample_with_lotgmp_enhanced(sampler, model, device: str, steps: int, x_size, cond, uncond, cfg_scale: float, progress: bool, use_lotgmp: bool, scale_path: str=None, shift_path: str=None, use_average: bool=False):
    sampler.make_schedule(steps)
    sampler.to(device)
    x_T = torch.randn(x_size, device=device, dtype=torch.float32)
    x = x_T
    timesteps = np.flip(sampler.timesteps)
    total_steps = len(sampler.timesteps)
    iterator = tqdm(timesteps, total=total_steps, disable=not progress)
    bs = x_size[0]
    if use_lotgmp and scale_path and shift_path:
        model.set_fixed_lotgmp_params(x_T, device, scale_path, shift_path, use_average)
    try:
        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = sampler.get_cfg_scale(cfg_scale, step)
            x = sampler.p_sample(model=model, x=x, model_t=model_t, t=t, cond=cond, uncond=uncond, cfg_scale=cur_cfg_scale)
    finally:
        if hasattr(model, '_fixed_scale_vector'):
            del model._fixed_scale_vector
        if hasattr(model, '_fixed_shift_vector'):
            del model._fixed_shift_vector
    return x

@torch.no_grad()
def main(args) -> None:
    device = torch.device('cuda:0')
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.inference.result_folder, exist_ok=True)
    cldm: ControlLDMWithLOTGMPEnhanced = ControlLDMWithLOTGMPEnhanced(unet_cfg=cfg.model.cldm.params.unet_cfg, vae_cfg=cfg.model.cldm.params.vae_cfg, clip_cfg=cfg.model.cldm.params.clip_cfg, controlnet_cfg=cfg.model.cldm.params.controlnet_cfg, latent_scale_factor=cfg.model.cldm.params.latent_scale_factor, lotgmp_cfg=get_lotgmp_config(cfg.model))
    sd = torch.load(cfg.inference.sd_path, map_location='cpu')['state_dict']
    unused, missing = cldm.load_pretrained_sd(sd)
    print(f'strictly load pretrained SD weight from {cfg.inference.sd_path}\nunused weights: {unused}\nmissing weights: {missing}')
    cldm.load_controlnet_from_ckpt(torch.load(cfg.inference.controlnet_path, map_location='cpu'))
    print(f'strictly load controlnet weight from checkpoint: {cfg.inference.controlnet_path}')
    lotgmp_path = get_cfg_value(cfg.inference, 'lotgmp_path', 'lpe_path')
    if lotgmp_path:
        cldm.load_lotgmp_weights(lotgmp_path)
    else:
        print('Warning: No LOTGMP weights provided.')
    lotgmp_modulation_path = get_cfg_value(cfg.inference, 'lotgmp_modulation_path', 'lpe_modulation_path')
    if lotgmp_modulation_path:
        cldm.load_lotgmp_modulation_weights(lotgmp_modulation_path)
    else:
        print('Warning: No LOTGMP modulation weights provided.')
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    cldm.eval().to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    rescaler = Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True)
    image_names = [os.path.basename(name) for ext in ('*.jpg', '*.jpeg', '*.png') for name in glob.glob(os.path.join(cfg.inference.image_folder, ext))]
    use_lotgmp = get_cfg_value(cfg.inference, 'use_lotgmp', 'use_lpe', True)
    scale_path = get_cfg_value(cfg.inference, 'lotgmp_scale_path', 'lpe_scale_path')
    shift_path = get_cfg_value(cfg.inference, 'lotgmp_shift_path', 'lpe_shift_path')
    use_average = get_cfg_value(cfg.inference, 'lotgmp_use_average', 'lpe_use_average', False)
    for image_name in tqdm(image_names):
        image = cv2.imread(os.path.join(cfg.inference.image_folder, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).unsqueeze(0)
        _, _, h, w = image.shape
        if h < 512 or w < 512:
            image = rescaler(image)
        _, _, h_, w_ = image.shape
        image = pad_to_multiples_of(image, multiple=64).to(device)
        prompt1 = 'a photograph with spatially varying PSF blur, optical aberrations, defocus, and chromatic fringing.'
        prompt2 = 'a photograph with spatially varying PSF blur, optical aberrations, defocus, chromatic fringing, and noticeable stray light with veiling glare.'
        cond = cldm.prepare_condition(image, [prompt1])
        cond2 = cldm.prepare_condition(image, [prompt2])
        uncond = {'c_img': torch.zeros_like(cond['c_img']), 'c_txt': copy.deepcopy(cond2['c_txt'])}
        z = sample_with_lotgmp_enhanced(sampler=sampler, model=cldm, device=device, steps=cfg.inference.get('steps', 10), x_size=cond['c_img'].shape, cond=cond, uncond=uncond, cfg_scale=cfg.inference.w, progress=False, use_lotgmp=use_lotgmp, scale_path=scale_path, shift_path=shift_path, use_average=use_average)
        result = (cldm.vae_decode(z) + 1) / 2
        result = result[:, :, :h_, :w_].clip(0.0, 1.0)
        result = Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)(result)
        result_tensor = result.squeeze(0)
        result_tensor = torch.clamp(result_tensor, 0, 1)
        result_pil = torchvision.transforms.ToPILImage()(result_tensor)
        suffix = '_lotgmp_film' if use_lotgmp else '_standard'
        output_name = f'{image_name[:-4]}{suffix}.jpg'
        result_pil.save(os.path.join(cfg.inference.result_folder, output_name), 'JPEG', quality=100, optimize=True, progressive=True)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
