import os
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy
from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
from diffbir.model.util import timestep_embedding
from diffbir.model.unet import ResBlock, TimestepEmbedSequential, TimestepBlock
from diffbir.model.attention import SpatialTransformer
import torch.nn as nn
import numpy as np

# LOTGMP is the latent-map predictor used by VeilGen.
# scale_vector represents the latent transmission map.
# shift_vector represents the latent glare map.

def get_lotgmp_config(model_cfg):
    if hasattr(model_cfg, 'lotgmp'):
        return model_cfg.lotgmp
    return model_cfg.lpe

def sample_with_lotgmp_enhanced(sampler, model, device: str, steps: int, x_size, cond, uncond, cfg_scale: float, progress: bool, use_lotgmp: bool, z_blur=None):
    sampler.make_schedule(steps)
    sampler.to(device)
    x_T = torch.randn(x_size, device=device, dtype=torch.float32)
    x = x_T
    timesteps = np.flip(sampler.timesteps)
    total_steps = len(sampler.timesteps)
    iterator = tqdm(timesteps, total=total_steps, disable=not progress)
    bs = x_size[0]
    model._current_z_blur = z_blur if use_lotgmp and z_blur is not None else None
    try:
        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = sampler.get_cfg_scale(cfg_scale, step)
            x = sampler.p_sample(model=model, x=x, model_t=model_t, t=t, cond=cond, uncond=uncond, cfg_scale=cur_cfg_scale)
    finally:
        model._current_z_blur = None
    return x

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

    def forward(self, x, hint, timesteps, context, lotgmp_params=None):
        if lotgmp_params is None:
            batch_size = x.shape[0]
            lotgmp_params = {'scale_vector': torch.zeros_like(x), 'shift_vector': torch.zeros_like(x), 'uncond_mask': torch.zeros(batch_size, dtype=torch.bool, device=x.device)}
        return self._forward_with_lotgmp(x, hint, timesteps, context, lotgmp_params)

    def _forward_with_lotgmp(self, x, hint, timesteps, context, lotgmp_params):
        scale_vector = lotgmp_params['scale_vector']
        shift_vector = lotgmp_params['shift_vector']
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
                    scale_resized = F.interpolate(scale_vector, size=(H, W), mode='bilinear', align_corners=False)
                    shift_resized = F.interpolate(shift_vector, size=(H, W), mode='bilinear', align_corners=False)
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

    def forward(self, x_noisy, t, cond):
        c_txt = cond['c_txt']
        c_img = cond['c_img']
        uncond_mask = torch.all(c_img == 0, dim=(1, 2, 3))
        has_uncond_samples = torch.any(uncond_mask)
        if has_uncond_samples:
            z_blur = getattr(self, '_current_z_blur', None)
            if z_blur is not None:
                lotgmp_output = self.lotgmp(x_noisy, z_blur, t)
                scale_vector = torch.where(uncond_mask.view(-1, 1, 1, 1), lotgmp_output['scale_vector'], torch.zeros_like(lotgmp_output['scale_vector']))
                shift_vector = torch.where(uncond_mask.view(-1, 1, 1, 1), lotgmp_output['shift_vector'], torch.zeros_like(lotgmp_output['shift_vector']))
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

def main(args) -> None:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f'Experiment directory created at {exp_dir}')
    cldm: ControlLDMWithLOTGMPEnhanced = ControlLDMWithLOTGMPEnhanced(unet_cfg=cfg.model.cldm.params.unet_cfg, vae_cfg=cfg.model.cldm.params.vae_cfg, clip_cfg=cfg.model.cldm.params.clip_cfg, controlnet_cfg=cfg.model.cldm.params.controlnet_cfg, latent_scale_factor=cfg.model.cldm.params.latent_scale_factor, lotgmp_cfg=get_lotgmp_config(cfg.model))
    sd = torch.load(cfg.train.sd_path, map_location='cpu')['state_dict']
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(f'strictly load pretrained SD weight from {cfg.train.sd_path}\nunused weights: {unused}\nmissing weights: {missing}')
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location='cpu'))
        if accelerator.is_main_process:
            print(f'strictly load controlnet weight from checkpoint: {cfg.train.resume}')
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(f'strictly load controlnet weight from pretrained SD\nweights initialized with newly added zeros: {init_with_new_zero}\nweights initialized from scratch: {init_with_scratch}')
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    controlnet_params = list(cldm.controlnet.parameters())
    lotgmp_params = list(cldm.lotgmp.parameters())
    lotgmp_modulation_params = list(cldm.controlnet_enhanced.lotgmp_modulation_layers.parameters())
    controlnet_opt = torch.optim.AdamW(controlnet_params, lr=cfg.train.learning_rate)
    lotgmp_lr = cfg.train.get('lotgmp_learning_rate', cfg.train.get('lpe_learning_rate', 0.0005))
    lotgmp_opt = torch.optim.AdamW(lotgmp_params + lotgmp_modulation_params, lr=lotgmp_lr)
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(dataset=dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    if accelerator.is_main_process:
        print(f'Dataset contains {len(dataset):,} images')
    cldm.train().to(device)
    diffusion.to(device)
    cldm, controlnet_opt, lotgmp_opt, loader = accelerator.prepare(cldm, controlnet_opt, lotgmp_opt, loader)
    pure_cldm: ControlLDMWithLOTGMPEnhanced = accelerator.unwrap_model(cldm)
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f'Training for {max_steps} steps...')
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_main_process, unit='batch', total=len(loader))
        for batch in loader:
            to(batch, device)
            gt, lq, prompt, idf = batch
            gt = gt.contiguous().float()
            lq = lq.contiguous().float()
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                cond = pure_cldm.prepare_condition(lq, prompt)
                for i in range(len(lq)):
                    if idf[i] == 'uncond':
                        cond['c_img'][i] = torch.zeros_like(cond['c_img'][i])
                cond['c_img'] = cond['c_img'].contiguous().float()
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
            noise = torch.randn_like(z_0)
            x_noisy = diffusion.q_sample(z_0, t, noise)
            pure_cldm._current_z_blur = z_0
            eps_pred = pure_cldm(x_noisy, t, cond)
            diffusion_loss = F.mse_loss(eps_pred, noise)
            pure_cldm._current_z_blur = None
            total_loss = diffusion_loss
            controlnet_opt.zero_grad()
            lotgmp_opt.zero_grad()
            accelerator.backward(total_loss)
            controlnet_opt.step()
            lotgmp_opt.step()
            accelerator.wait_for_everyone()
            global_step += 1
            step_loss.append(diffusion_loss.item())
            epoch_loss.append(diffusion_loss.item())
            pbar.update(1)
            pbar.set_description(f'Epoch: {epoch:04d}, Global Step: {global_step:07d}, Enhanced LOTGMP Loss: {diffusion_loss.item():.6f}')
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                avg_diffusion_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar('loss/enhanced_lotgmp_loss_step', avg_diffusion_loss, global_step)
                    writer.add_scalar('loss/total_loss_step', avg_diffusion_loss, global_step)
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    controlnet_checkpoint = pure_cldm.controlnet.state_dict()
                    controlnet_ckpt_path = f'{ckpt_dir}/controlnet_{global_step:07d}.pt'
                    torch.save(controlnet_checkpoint, controlnet_ckpt_path)
                    lotgmp_checkpoint = pure_cldm.lotgmp.state_dict()
                    lotgmp_ckpt_path = f'{ckpt_dir}/lotgmp_{global_step:07d}.pt'
                    torch.save(lotgmp_checkpoint, lotgmp_ckpt_path)
                    lotgmp_modulation_checkpoint = pure_cldm.controlnet_enhanced.lotgmp_modulation_layers.state_dict()
                    lotgmp_modulation_ckpt_path = f'{ckpt_dir}/lotgmp_modulation_{global_step:07d}.pt'
                    torch.save(lotgmp_modulation_checkpoint, lotgmp_modulation_ckpt_path)
            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_gt, log_lq = (gt[:N], lq[:N])
                log_prompt = prompt[:N]
                log_idf = idf[:N]
                cldm.eval()
                with torch.no_grad():
                    z_enhanced = sample_with_lotgmp_enhanced(sampler=sampler, model=pure_cldm, device=device, steps=50, x_size=(len(log_gt), *z_0.shape[1:]), cond=log_cond, uncond=None, cfg_scale=1.0, progress=accelerator.is_main_process, use_lotgmp=True, z_blur=z_0[:N])
                    prompt1 = 'a photograph with spatially varying PSF blur, optical aberrations, defocus, and chromatic fringing.'
                    prompt2 = 'a photograph with spatially varying PSF blur, optical aberrations, defocus, chromatic fringing, and noticeable stray light with veiling glare.'
                    log_cond1 = pure_cldm.prepare_condition(log_lq, [prompt1] * len(log_lq))
                    log_cond2 = pure_cldm.prepare_condition(log_lq, [prompt2] * len(log_lq))
                    log_uncond = {'c_img': torch.zeros_like(log_cond1['c_img']), 'c_txt': copy.deepcopy(log_cond2['c_txt'])}
                    z_cfg = sample_with_lotgmp_enhanced(sampler=sampler, model=pure_cldm, device=device, steps=50, x_size=(len(log_gt), *z_0.shape[1:]), cond=log_cond1, uncond=log_uncond, cfg_scale=0.85, progress=accelerator.is_main_process, use_lotgmp=True, z_blur=z_0[:N])
                    if accelerator.is_main_process:
                        for tag, image in [('image/samples_enhanced_lotgmp', (pure_cldm.vae_decode(z_enhanced) + 1) / 2), ('image/samples_cfg', (pure_cldm.vae_decode(z_cfg) + 1) / 2), ('image/gt', (log_gt + 1) / 2), ('image/lq', log_lq), ('image/condition_decoded', (pure_cldm.vae_decode(log_cond['c_img']) + 1) / 2), ('image/prompt_original', (log_txt_as_img((512, 512), log_prompt) + 1) / 2), ('image/prompt1_cfg', (log_txt_as_img((512, 512), [prompt1] * len(log_lq)) + 1) / 2), ('image/prompt2_cfg', (log_txt_as_img((512, 512), [prompt2] * len(log_lq)) + 1) / 2)]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar('loss/enhanced_lotgmp_loss_epoch', avg_epoch_loss, global_step)
    if accelerator.is_main_process:
        print('done!')
        writer.close()
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args)
