"""
No-Reference Image Quality Assessment (NR-IQA) metrics for BasicSR.

This module provides implementations of various NR-IQA metrics using pyiqa library:
- CLIPIQA: CLIP-based image quality assessment
- NIQE: Natural Image Quality Evaluator (traditionally uses Y-channel)
- MANIQA: Multi-dimension Attention Network for IQA
- MUSIQ: Multi-scale Image Quality Transformer

Note on Y-channel usage:
- NIQE: Benefits from Y-channel conversion (test_y_channel=True recommended)
- CLIPIQA, MANIQA, MUSIQ: Designed for RGB images (test_y_channel=False recommended)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_clipiqa(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate CLIPIQA (no-reference image quality assessment).

    Note: CLIPIQA is designed to work on RGB images and doesn't benefit from Y-channel conversion.
    The test_y_channel parameter is kept for compatibility but has minimal impact.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Reference images with range [0, 255] (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False (not recommended for CLIPIQA).

    Returns:
        float: CLIPIQA result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    # CLIPIQA works best on RGB images, Y-channel conversion is not recommended
    if test_y_channel:
        img = to_y_channel(img)
        # to_y_channel returns (H, W, 1), we need (H, W) for Y channel
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        # Convert BGR to RGB for RGB mode
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Initialize CLIPIQA metric
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
    
    # Convert to tensor format [1, C, H, W] with range [0, 1]
    if test_y_channel:
        # For Y channel: img_pil is 2D (H, W), need to make it 4D [1, 1, H, W]
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        # For RGB: img_pil is 3D (H, W, C), need to make it 4D [1, C, H, W]
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        score = clipiqa_metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_niqe(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate NIQE (Natural Image Quality Evaluator).

    Note: NIQE traditionally works on Y channel of YCbCr and benefits from Y-channel conversion.
    Setting test_y_channel=True is recommended for NIQE.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Reference images with range [0, 255] (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False (recommended: True for NIQE).

    Returns:
        float: NIQE result (lower is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img = to_y_channel(img)
        # to_y_channel returns (H, W, 1), we need (H, W) for Y channel
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        # Convert BGR to RGB for RGB mode
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Initialize NIQE metric
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    
    # Convert to tensor format [1, C, H, W] with range [0, 1]
    if test_y_channel:
        # For Y channel: img_pil is 2D (H, W), need to make it 4D [1, 1, H, W]
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        # For RGB: img_pil is 3D (H, W, C), need to make it 4D [1, C, H, W]
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        score = niqe_metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_maniqa(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate MANIQA (Multi-dimension Attention Network for No-Reference Image Quality Assessment).

    Note: MANIQA is a deep learning-based method designed for RGB images.
    Y-channel conversion is not recommended as it may reduce performance.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Reference images with range [0, 255] (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False (not recommended for MANIQA).

    Returns:
        float: MANIQA result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img = to_y_channel(img)
        # to_y_channel returns (H, W, 1), we need (H, W) for Y channel
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        # Convert BGR to RGB for RGB mode
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Initialize MANIQA metric
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    
    # Convert to tensor format [1, C, H, W] with range [0, 1]
    if test_y_channel:
        # For Y channel: img_pil is 2D (H, W), need to make it 4D [1, 1, H, W]
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        # For RGB: img_pil is 3D (H, W, C), need to make it 4D [1, C, H, W]
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        score = maniqa_metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_musiq(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate MUSIQ (Multi-scale Image Quality Transformer).

    Note: MUSIQ is a transformer-based method designed for RGB images.
    Y-channel conversion is not recommended as it may reduce performance.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Reference images with range [0, 255] (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False (not recommended for MUSIQ).

    Returns:
        float: MUSIQ result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    
    img = reorder_image(img, input_order=input_order)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if test_y_channel:
        img = to_y_channel(img)
        # to_y_channel returns (H, W, 1), we need (H, W) for Y channel
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        # Convert BGR to RGB for RGB mode
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    # Initialize MUSIQ metric
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    
    # Convert to tensor format [1, C, H, W] with range [0, 1]
    if test_y_channel:
        # For Y channel: img_pil is 2D (H, W), need to make it 4D [1, 1, H, W]
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        # For RGB: img_pil is 3D (H, W, C), need to make it 4D [1, C, H, W]
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        score = musiq_metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


# PyTorch versions for better GPU performance
@METRIC_REGISTRY.register()
def calculate_clipiqa_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate CLIPIQA (PyTorch version).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Reference images with range [0, 1], shape (n, 3/1, h, w) (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: CLIPIQA result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
    
    device = img.device
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=device)
    
    with torch.no_grad():
        score = clipiqa_metric(img)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_niqe_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate NIQE (PyTorch version).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Reference images with range [0, 1], shape (n, 3/1, h, w) (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: NIQE result (lower is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
    
    device = img.device
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    
    with torch.no_grad():
        score = niqe_metric(img)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_maniqa_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate MANIQA (PyTorch version).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Reference images with range [0, 1], shape (n, 3/1, h, w) (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: MANIQA result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
    
    device = img.device
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    
    with torch.no_grad():
        score = maniqa_metric(img)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_musiq_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """Calculate MUSIQ (PyTorch version).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Reference images with range [0, 1], shape (n, 3/1, h, w) (not used for no-reference metrics).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: MUSIQ result (higher is better).
    """
    # For no-reference metrics, we only use img (the input image)
    # img2 is ignored but kept for compatibility with the framework
    
    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
    
    device = img.device
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    
    with torch.no_grad():
        score = musiq_metric(img)
        score_val = float(score.item() if hasattr(score, 'item') else score)
    
    return score_val


@METRIC_REGISTRY.register()
def calculate_qalign(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate QAlign (no-reference IQA via pyiqa).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: QAlign score (higher is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('qalign', device=device)

    if test_y_channel:
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val


@METRIC_REGISTRY.register()
def calculate_liqe(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LIQE (no-reference IQA via pyiqa).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: LIQE score (higher is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('liqe', device=device)

    if test_y_channel:
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val


@METRIC_REGISTRY.register()
def calculate_brisque(img, img2, crop_border, input_order='HWC', test_y_channel=True, **kwargs):
    """Calculate BRISQUE (no-reference IQA via pyiqa).

    Note: BRISQUE traditionally works on Y channel; default test_y_channel=True.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: BRISQUE score (lower is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('brisque', device=device)

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val


@METRIC_REGISTRY.register()
def calculate_nima(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate NIMA (Neural Image Assessment) via pyiqa.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: NIMA score (higher is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('nima', device=device)

    if test_y_channel:
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val


@METRIC_REGISTRY.register()
def calculate_pi(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PI (Perceptual Index) via pyiqa.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: PI score (lower is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('pi', device=device)

    if test_y_channel:
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val


@METRIC_REGISTRY.register()
def calculate_piqe(img, img2, crop_border, input_order='HWC', test_y_channel=True, **kwargs):
    """Calculate PIQE (Perception based Image Quality Evaluator) via pyiqa.

    Note: PIQE traditionally works on Y channel; default test_y_channel=True.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Unused (compatibility only).
        crop_border (int): Cropped pixels in each edge of an image.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): If True, convert to Y channel before eval.
    Returns:
        float: PIQE score (lower is better).
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img_pil = img.squeeze(-1).astype(np.uint8)
        img_tensor = torch.from_numpy(img_pil).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        img_pil = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_pil).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('piqe', device=device)

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score = metric(img_tensor)
        score_val = float(score.item() if hasattr(score, 'item') else score)

    return score_val
