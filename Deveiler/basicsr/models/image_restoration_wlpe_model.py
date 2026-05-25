import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, imwrite_jpg
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class ImageRestorationModel_WLPE(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(ImageRestorationModel_WLPE, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    # def setup_optimizers(self):
    #     train_opt = self.opt['train']
    #     optim_params = []
    #     for k, v in self.net_g.named_parameters():
    #         if v.requires_grad:
    #             optim_params.append(v)
    #         else:
    #             logger = get_root_logger()
    #             logger.warning(f'Params {k} will not be optimized.')

    #     optim_type = train_opt['optim_g'].pop('type')
    #     self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
    #     self.optimizers.append(self.optimizer_g)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'scale_map' in data:
            self.scale_map = data['scale_map'].to(self.device)
        if 'shift_map' in data:
            self.shift_map = data['shift_map'].to(self.device)
        self.origin_scale_map = self.scale_map
        self.origin_shift_map = self.shift_map
    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t


    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.scale_map = self.origin_scale_map
        self.shift_map = self.origin_shift_map

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # self.output = self.net_g(self.lq)
        preds = self.net_g(self.lq,self.scale_map,self.shift_map)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                
                # In validation phase, LPE maps must be available
                if not hasattr(self, 'scale_map') or self.scale_map is None:
                    raise ValueError("VALIDATION ERROR: scale_map is required but not available")
                if not hasattr(self, 'shift_map') or self.shift_map is None:
                    raise ValueError("VALIDATION ERROR: shift_map is required but not available")
                
                pred = self.net_g(self.lq[i:j, :, :, :], self.scale_map[i:j, :, :, :], self.shift_map[i:j, :, :, :])
                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def single_image_inference(self, img, save_path):
        self.feed_data(data={'lq': img.unsqueeze(dim=0)})

        if self.opt['val'].get('grids', False):
            self.grids()

        self.test()

        if self.opt['val'].get('grids', False):
            self.grids_inverse()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    # def test(self):
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g_ema(self.lq)
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             self.output = self.net_g(self.lq)
    #         self.net_g.train()

    # def test_selfensemble(self):
    #     # TODO: to be tested
    #     # 8 augmentations
    #     # modified from https://github.com/thstkdgus35/EDSR-PyTorch

    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == 'v':
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #         elif op == 'h':
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #         elif op == 't':
    #             tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()

    #         return ret

    #     # prepare augmented data
    #     lq_list = [self.lq]
    #     for tf in 'v', 'h', 't':
    #         lq_list.extend([_transform(t, tf) for t in lq_list])

    #     # inference
    #     if hasattr(self, 'net_g_ema'):
    #         self.net_g_ema.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #     else:
    #         self.net_g.eval()
    #         with torch.no_grad():
    #             out_list = [self.net_g_ema(aug) for aug in lq_list]
    #         self.net_g.train()

    #     # merge results
    #     for i in range(len(out_list)):
    #         if i > 3:
    #             out_list[i] = _transform(out_list[i], 't')
    #         if i % 4 > 1:
    #             out_list[i] = _transform(out_list[i], 'h')
    #         if (i % 4) % 2 == 1:
    #             out_list[i] = _transform(out_list[i], 'v')
    #     output = torch.cat(out_list, dim=0)

    #     self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            if self.opt['val'].get('patch_test', False):
                self.test_patch()
            else:
                if self.opt['val'].get('grids', False):
                    self.grids()
                self.test()

                if self.opt['val'].get('grids', False):
                    self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            #只取visuals['lq']前三个通道
            lq_img = tensor2img([visuals['lq'][:, :3, :, :]])

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                 f'{img_name}.jpg')
                imwrite_jpg(sr_img, save_img_path)
                
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def test_patch(self):
        """Process image in overlapping patches with confidence map based fusion.
        The confidence map ensures smooth blending in overlapping regions.
        Includes padding to handle images that don't divide evenly by patch size.
        """
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1  # 确保batch size为1

        # 保存原始图像
        self.origin_lq = self.lq.clone()

        # Get patch size and overlap from config
        patch_size = self.opt['val'].get('patch_size', 256)
        overlap = self.opt['val'].get('overlap', 32)

        # Calculate required padding
        stride = patch_size - overlap
        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride

        # Pad image if necessary
        if pad_h > 0 or pad_w > 0:
            self.lq = torch.nn.functional.pad(self.lq, (0, pad_w, 0, pad_h), mode='reflect')
            h_padded, w_padded = self.lq.shape[2:]
        else:
            h_padded, w_padded = h, w

        # Calculate number of patches
        num_row = (h_padded - patch_size) // stride + 1
        num_col = (w_padded - patch_size) // stride + 1

        # Initialize confidence map and output tensor
        confidence_map = torch.zeros((b, 1, h_padded, w_padded)).to(self.device)
        output = torch.zeros((b, c, h_padded, w_padded)).to(self.device)

        # Process each patch
        for i in range(num_row):
            for j in range(num_col):
                # Calculate patch coordinates
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + patch_size
                end_j = start_j + patch_size

                # Extract patch
                patch = self.lq[:, :, start_i:end_i, start_j:end_j]

                # Generate confidence map for this patch
                patch_confidence = torch.ones((b, 1, patch_size, patch_size)).to(self.device)

                # Apply confidence reduction in overlapping regions
                if overlap > 0:
                    # 创建水平和垂直的权重
                    h_weights = torch.ones(patch_size).to(self.device)
                    v_weights = torch.ones(patch_size).to(self.device)

                    # 计算水平权重
                    if j > 0:  # 左重叠
                        h_weights[:overlap] = torch.linspace(0, 1, overlap).to(self.device)
                    if j < num_col - 1:  # 右重叠
                        h_weights[-overlap:] = torch.linspace(1, 0, overlap).to(self.device)

                    # 计算垂直权重
                    if i > 0:  # 上重叠
                        v_weights[:overlap] = torch.linspace(0, 1, overlap).to(self.device)
                    if i < num_row - 1:  # 下重叠
                        v_weights[-overlap:] = torch.linspace(1, 0, overlap).to(self.device)

                    # 将水平和垂直权重组合
                    h_weights = h_weights.view(1, 1, 1, -1)
                    v_weights = v_weights.view(1, 1, -1, 1)
                    patch_confidence = h_weights * v_weights

                # Process patch through network
                with torch.no_grad():
                    patch_output = self.net_g(patch)
                    if isinstance(patch_output, list):
                        patch_output = patch_output[-1]

                # Add weighted patch to output
                output[:, :, start_i:end_i, start_j:end_j] += patch_output * patch_confidence
                confidence_map[:, :, start_i:end_i, start_j:end_j] += patch_confidence

        # Verify confidence map
        confidence_sum = confidence_map.sum(dim=0, keepdim=True)
        print('min confidence sum: ', confidence_sum.min().item())
        print('max confidence sum: ', confidence_sum.max().item())
        if torch.any(torch.abs(confidence_sum - 1.0) > 1e-6):
            print("Warning: Confidence map weights do not sum to 1.0 in some regions")
            print(f"Min weight: {confidence_sum.min().item()}, Max weight: {confidence_sum.max().item()}")

        # Normalize output by confidence map
        output = output / confidence_map

        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        self.output = output
        self.lq = self.origin_lq
