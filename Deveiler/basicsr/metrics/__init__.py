from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
# from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_rmse, calculate_lpips,calculate_abs_rel,calculate_d1,calculate_d2,calculate_d3
from .nr_iqa_metric import calculate_niqe,calculate_maniqa,calculate_musiq,calculate_clipiqa,calculate_musiq_pt,calculate_clipiqa_pt,calculate_niqe_pt,calculate_maniqa_pt
# __all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips','calculate_rmse','calculate_abs_rel','calculate_d1','calculate_d2','calculate_d3','calculate_maniqa','calculate_musiq','calculate_clipiqa','calculate_musiq_pt','calculate_clipiqa_pt','calculate_niqe_pt','calculate_maniqa_pt']
def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
