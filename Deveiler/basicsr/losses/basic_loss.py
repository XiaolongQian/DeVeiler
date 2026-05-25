import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


def get_dark_channel(I, w):
    
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc
    
@LOSS_REGISTRY.register()
class Low_Freq_Loss(nn.Module):
    """Glare-only spectral loss (low-frequency magnitude constraint).

    Implements:
        L = || (|F_pred| ⊙ Mask_L) - (|F_target| ⊙ Mask_L) ||_1

    Where Mask_L is a centered low-frequency rectangle controlled by beta.
    """

    def __init__(self, loss_weight=0.1, beta=0.1, eps=1e-12):
        super(Low_Freq_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.beta = float(beta) #--> 控制低频Mask_L的大小
        self.eps = float(eps)

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f'pred and target must have the same shape, got {pred.shape} vs {target.shape}')
        if pred.dim() != 4:
            raise ValueError(f'Expected pred/target in NCHW format, got dim={pred.dim()}')
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f'beta must be in (0, 1], got {self.beta}')

        # AMP safety: do FFT & spectrum math in FP32 to avoid FP16 overflow (e.g. large DC term).
        pred_f = pred.float()
        target_f = target.float()

        n, c, h, w = pred_f.shape
        low_h = max(1, int(round(h * self.beta)))
        low_w = max(1, int(round(w * self.beta)))
        y0 = (h - low_h) // 2
        x0 = (w - low_w) // 2

        # 2D FFT -> shift DC to center -> magnitude spectrum
        pred_fft = torch.fft.fft2(pred_f, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.fft2(target_f, dim=(-2, -1), norm='ortho')

        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

        # Stable magnitude: |a+jb| = sqrt(a^2 + b^2 + eps)
        pred_mag = torch.sqrt(pred_fft.real.square() + pred_fft.imag.square() + self.eps)
        target_mag = torch.sqrt(target_fft.real.square() + target_fft.imag.square() + self.eps)

        # Center low-frequency crop (Mask_L)
        pred_low = pred_mag[:, :, y0:y0 + low_h, x0:x0 + low_w]
        target_low = target_mag[:, :, y0:y0 + low_h, x0:x0 + low_w]

        # L1 distance on low-frequency magnitudes only
        loss = F.l1_loss(pred_low, target_low, reduction='mean')
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class SWT_Loss(nn.Module):
    """
    单层平稳小波解耦损失 (Stationary Wavelet Transform Loss)
    使用 Stride=1 的 Haar 卷积核，保留空间位置，减少网格伪影风险。
    """

    def __init__(self, loss_weight=0.1 ,alpha=0.5, beta=2.0):
        super(SWT_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # 低频 LL (宏观眩光/亮度) 的权重
        self.beta = beta  # 高频 LH, HL, HH (像差边缘/结构) 的权重

        # 构造 Haar 小波的四个滤波器: LL, LH(水平), HL(垂直), HH(对角线)
        # 乘以 0.5 是为了能量归一化
        haar_filters = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]]],  # LL: 低通滤波 (求均值)
                [[[-0.5, -0.5], [0.5, 0.5]]],  # LH: 水平高频边缘
                [[[-0.5, 0.5], [-0.5, 0.5]]],  # HL: 垂直高频边缘
                [[[0.5, -0.5], [-0.5, 0.5]]],  # HH: 对角线高频边缘
            ],
            dtype=torch.float32,
        )

        # 输入图像通常是 RGB 3通道，使用分组卷积(groups=3)使其对每个通道独立运算
        # 因此需要把滤波器复制 3 份: [4,1,2,2] -> [12,1,2,2]
        self.register_buffer('weight', haar_filters.repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        """pred, target: [Batch, 3, H, W]."""
        if pred.shape != target.shape:
            raise ValueError('Pred and Target must have the same shape.')

        # 使用 stride=1 + padding='same' 保证分辨率不变，实现单层 SWT
        pred_swt = F.conv2d(pred, self.weight, stride=1, padding='same', groups=3)
        target_swt = F.conv2d(target, self.weight, stride=1, padding='same', groups=3)

        # 输出通道数为 12 (3个颜色通道 * 4个小波子带)
        # Reshape 为 [Batch, 3(RGB), 4(Subbands), H, W] 方便分离
        b, _, h_out, w_out = pred_swt.shape
        pred_swt = pred_swt.view(b, 3, 4, h_out, w_out)
        target_swt = target_swt.view(b, 3, 4, h_out, w_out)

        # 提取 LL 子带 (索引 0)
        pred_ll = pred_swt[:, :, 0, :, :]
        target_ll = target_swt[:, :, 0, :, :]

        # 提取高频子带 LH, HL, HH (索引 1, 2, 3)
        pred_high = pred_swt[:, :, 1:, :, :]
        target_high = target_swt[:, :, 1:, :, :]

        # 分别计算 L1 Loss
        loss_ll = F.l1_loss(pred_ll, target_ll)
        loss_high = F.l1_loss(pred_high, target_high)

        return self.loss_weight * self.alpha * loss_ll + self.loss_weight * self.beta * loss_high


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # 如果你的框架有自动注册器，保留你的 @LOSS_REGISTRY.register()
@LOSS_REGISTRY.register()
class MultiScaleDWTLoss(nn.Module):
    """
    多尺度离散小波解耦损失 (Multi-Scale DWT Loss)
    使用 Stride=2 进行真正的小波下采样。
    极大地降低显存占用，同时通过多级金字塔获得巨大的感受野，完美剥离宏观眩光。
    """
    def __init__(self, loss_weight=0.1, alpha=0.5, beta=2.0, num_levels=3):
        super(MultiScaleDWTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # 低频 LL (宏观眩光/亮度) 的权重
        self.beta = beta    # 高频 LH, HL, HH (像差边缘/结构) 的权重
        self.num_levels = num_levels # 分解层数，推荐 3 或 4

        # 构造 Haar 小波滤波器
        haar_filters = torch.tensor(
            [
                [[[0.5, 0.5], [0.5, 0.5]]],  # LL: 低通滤波
                [[[-0.5, -0.5], [0.5, 0.5]]],  # LH: 水平高频边缘
                [[[-0.5, 0.5], [-0.5, 0.5]]],  # HL: 垂直高频边缘
                [[[0.5, -0.5], [-0.5, 0.5]]],  # HH: 对角线高频边缘
            ],
            dtype=torch.float32,
        )

        # 复制 3 份以适应 RGB 通道的 Group 卷积: [12,1,2,2]
        self.register_buffer('weight', haar_filters.repeat(3, 1, 1, 1))

    def _dwt_once(self, x):
        """执行单层 DWT (注意 stride=2 下采样)"""
        # 为了防止输入图像长宽是奇数导致 stride=2 报错，动态 pad 到偶数
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        # stride=2 带来真正的空间下采样！
        out = F.conv2d(x, self.weight, stride=2, padding=0, groups=3)
        return out

    def forward(self, pred, target):
        """pred, target: [Batch, 3, H, W]."""
        if pred.shape != target.shape:
            raise ValueError('Pred and Target must have the same shape.')

        curr_pred = pred
        curr_gt = target
        
        loss_high = 0.0
        
        # 逐层下潜 (Multi-scale 金字塔)
        for level in range(self.num_levels):
            pred_dwt = self._dwt_once(curr_pred)
            gt_dwt = self._dwt_once(curr_gt)
            
            b, _, h_out, w_out = pred_dwt.shape
            pred_dwt = pred_dwt.view(b, 3, 4, h_out, w_out)
            gt_dwt = gt_dwt.view(b, 3, 4, h_out, w_out)
            
            # 提取 LL (索引 0) 作为下一层的输入
            curr_pred = pred_dwt[:, :, 0, :, :]
            curr_gt = gt_dwt[:, :, 0, :, :]
            
            # 提取当前层的高频 (LH, HL, HH) 并累计 Loss
            pred_high = pred_dwt[:, :, 1:, :, :]
            gt_high = gt_dwt[:, :, 1:, :, :]
            
            # 由于高频系数逐层缩小，直接累加即可 (或按层加权)
            loss_high += F.l1_loss(pred_high, gt_high)

        # 经过 num_levels 下潜后，curr_pred 已经是最深层的宏观光幕图
        # 计算极深层 LL 的误差
        loss_ll = F.l1_loss(curr_pred, curr_gt)
        
        # 高频误差求平均，防止层数带来的量级膨胀
        loss_high = loss_high / self.num_levels

        return self.loss_weight * (self.alpha * loss_ll + self.beta * loss_high)


from pytorch_wavelets import DWTForward
@LOSS_REGISTRY.register()
class DWT_Pyramid_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, levels=4, alpha=0.5, beta=2.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        # J=4 表示下潜 4 层。
        self.dwt = DWTForward(J=levels, wave='haar', mode='reflect')

    def forward(self, pred, gt):
        """
        输入 pred 和 gt 的 Shape: [B, 3, 1280, 1920]
        """
        
        # =====================================================================
        # 核心拆解：p_ll 和 p_highs 到底存了什么？
        # =====================================================================
        p_ll, p_highs = self.dwt(pred)
        g_ll, g_highs = self.dwt(gt)

        # 🎯 p_ll (极深层低频近似)
        # Shape: [B, 3, 80, 120]   (因为下采样了 4 次：1280/(2^4) = 80, 1920/(2^4) = 120)
        # 物理意义：这是扒掉了 4 层皮之后，剩下的最纯粹的 Veiling Glare (漫反射光幕)。
        # 它极其模糊，感受野高达 16x16，绝大部分像差轮廓已经被洗掉。
        loss_ll = F.l1_loss(p_ll, g_ll)

        # 🎯 p_highs (多尺度高频金字塔)
        # 类型：这是一个长度为 4 的 List (列表)。
        # p_highs = [Level_1_高频, Level_2_高频, Level_3_高频, Level_4_高频]
        
        # 我们来看看这个 List 里面每一个元素的 Shape 和物理意义：
        
        # p_highs[0] (第一层高频): 
        # Shape: [B, 3, 3, 640, 960] 
        # 解释：第2个 '3' 代表颜色RGB；第3个 '3' 代表三个方向(LH水平, HL垂直, HH对角)。
        # 物理意义：极细微的像差边缘、传感器噪点、极其锐利的文字边缘。
        
        # p_highs[1] (第二层高频): 
        # Shape: [B, 3, 3, 320, 480]
        # 物理意义：稍微粗一点的树枝、次级像差重影。
        
        # p_highs[2] (第三层高频): 
        # Shape: [B, 3, 3, 160, 240]
        # 物理意义：更粗的物体轮廓。
        
        # p_highs[3] (第四层高频): 
        # Shape: [B, 3, 3, 80, 120]
        # 物理意义：大块物体的边界走势。

        # =====================================================================
        # 计算高频 Loss
        # =====================================================================
        loss_high = 0.0
        
        # 这个 zip 会同时遍历预测图和GT图的 4 个高频层级
        for p_h, g_h in zip(p_highs, g_highs):
            # 每次循环，p_h 的 Shape 会依次是：
            # [B, 3, 3, 640, 960] -> [B, 3, 3, 320, 480] -> [B, 3, 3, 160, 240] -> [B, 3, 3, 80, 120]
            # 我们直接把这三个方向 (LH, HL, HH) 一锅端算 L1 误差，逼迫网络全方位保底边缘。
            loss_high += F.l1_loss(p_h, g_h)
            
        # 求平均，防止层数太多导致 Loss 数值爆炸
        loss_high = loss_high / len(p_highs)

        return self.loss_weight * (self.alpha * loss_ll + self.beta * loss_high)

@LOSS_REGISTRY.register()
class AIF_Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(AIF_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # 计算预测图像的暗通道
        ab_pred = get_dark_channel(pred, 15)
        # print(ab_pred.shape)
        #[B,1,H,W] -> [B,3,H,W]
        rgb_ab_pred = ab_pred.repeat(1,3,1,1)
        #根据像差图像的暗通道，计算AIF_mask
        #暗通道
        AIF_mask  = 1 - rgb_ab_pred

        #保证AIF_mask的值在0-1之间
        AIF_mask = 1 + torch.clamp(AIF_mask, 0, 1)
        # print(AIF_mask.shape)
        #计算AIF_loss
        AIF_loss = self.loss_weight * l1_loss(AIF_mask * pred, AIF_mask * target, reduction='mean')
        return AIF_loss

@LOSS_REGISTRY.register()
class DIF_AIF_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, mask_weight=2.0, normalize_mask=True):
        super(DIF_AIF_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # 计算预测图像的暗通道
        ab_dark_channel = get_dark_channel(pred, 15)
        gt_dark_channel = get_dark_channel(target, 15)

        #计算dif_dark_channel
        dif_dark_channel = torch.abs(ab_dark_channel - gt_dark_channel)
        #max
        max_val = dif_dark_channel.max()
        norm_dif = dif_dark_channel / (max_val + 1e-6) # 1e-6防止除0
        # print(dif_dark_channel.min(), dif_dark_channel.max())
        # print(ab_pred.shape)
        #[B,1,H,W] -> [B,3,H,W]
        AIF_mask = norm_dif.repeat(1,3,1,1)
        #根据像差图像的暗通道，计算AIF_mask

        # #保证AIF_mask的值在0-1之间
        # AIF_mask = 1 + torch.clamp(AIF_mask, 0, 1)
        # print(AIF_mask.shape)
        #计算AIF_loss
        AIF_loss = self.loss_weight * l1_loss(AIF_mask * pred, AIF_mask * target, reduction='mean')
        return AIF_loss
@LOSS_REGISTRY.register()
class DIF_DCP_Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DIF_DCP_Loss, self).__init__()
        self.loss_weight = loss_weight
    def forward(self, pred, target):
        # 计算预测图像的暗通道
        ab_dark_channel = get_dark_channel(pred, 15)
        gt_dark_channel = get_dark_channel(target, 15)
        #计算差异
        dif_dark_channel = torch.abs(ab_dark_channel - gt_dark_channel)

        # mask = dif_dark_channel / (dif_dark_channel.max() + 1e-6)
        #计算损失
        DCP_loss = self.loss_weight * l1_loss(dif_dark_channel, torch.zeros_like(dif_dark_channel), reduction='mean')

        return DCP_loss


@LOSS_REGISTRY.register()
class Supervised_DarkChannelLoss(nn.Module):
    def __init__(self, window_size=15, loss_weight=1.0):
        super(Supervised_DarkChannelLoss, self).__init__()
        self.window_size = window_size
        self.weight = loss_weight

    def forward(self, dehazed_image, gt_image):
        # 计算去雾图像的暗通道
        dehazed_dark_channel = get_dark_channel(dehazed_image, self.window_size)
        gt_dark_channel = get_dark_channel(gt_image, self.window_size)

        # 计算暗通道损失，最小化暗通道值

        # 根据权重缩放损失
        scaled_loss = self.weight * l1_loss(dehazed_dark_channel, gt_dark_channel, None, reduction='mean')

        return scaled_loss

@LOSS_REGISTRY.register()
class DarkChannelLoss(nn.Module):
    def __init__(self, window_size=15, loss_weight=1.0):
        super(DarkChannelLoss, self).__init__()
        self.window_size = window_size
        self.weight = loss_weight

    def forward(self, dehazed_image):
        # 计算去雾图像的暗通道
        dehazed_dark_channel = get_dark_channel(dehazed_image, self.window_size)

        # 计算暗通道损失，最小化暗通道值
        dark_channel_loss = torch.mean(dehazed_dark_channel)

        # 根据权重缩放损失
        scaled_loss = self.weight * dark_channel_loss

        return scaled_loss
    
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)



import pyiqa
@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """LPIPS loss with vgg backbone.
    """
    def __init__(self, loss_weight = 1.0):
        super(LPIPSLoss, self).__init__()
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        return self.model(x, gt) * self.loss_weight, None

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

@LOSS_REGISTRY.register()
class silog_loss(nn.Module):
    def __init__(self, loss_weight = 1,variance_focus=0.85):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
    # def forward(self, depth_est, depth_gt, mask):
    #     d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
    #     return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

    def forward(self, pred, target):
        mask = target > 0.1
        d = torch.log(pred[mask]) - torch.log(target[mask])
        return self.loss_weight * torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

@LOSS_REGISTRY.register()
class FuseLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FuseLoss, self).__init__()
        self.irloss = L1Loss(loss_weight, reduction)
        self.x1hatloss = L1Loss(loss_weight, reduction)

    def forward(self, pred, x1hat, target):
        Lir = self.irloss.forward(pred, target)
        Lx1hat = self.x1hatloss.forward(x1hat, target)

        L = Lir + 0.5*Lx1hat

        return L


@LOSS_REGISTRY.register()
class fft_l1loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(fft_l1loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm=None)
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm=None)

        return self.loss_weight * l1_loss(pred_fft, target_fft, reduction=self.reduction)

@LOSS_REGISTRY.register()
class Improved_DIF_AIF_Loss(nn.Module):
    """改进的DIF_AIF_Loss，更好的mask设计和归一化策略"""
    def __init__(self, loss_weight=1.0, mask_weight=3.0, normalize_method='global', smooth_mask=True):
        super(Improved_DIF_AIF_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.mask_weight = mask_weight  # 控制mask的最大权重
        self.normalize_method = normalize_method  # 'global', 'local', 'none'
        self.smooth_mask = smooth_mask  # 是否对mask进行平滑

    def forward(self, pred, target):
        # 计算预测图像和GT的暗通道
        pred_dark_channel = get_dark_channel(pred, 15)
        gt_dark_channel = get_dark_channel(target, 15)

        # 计算暗通道差异
        dif_dark_channel = torch.abs(pred_dark_channel - gt_dark_channel)
        
        # 归一化差异值
        if self.normalize_method == 'global':
            # 全局归一化
            dif_min = dif_dark_channel.min()
            dif_max = dif_dark_channel.max()
            if dif_max > dif_min:
                dif_dark_channel = (dif_dark_channel - dif_min) / (dif_max - dif_min)
            else:
                dif_dark_channel = torch.zeros_like(dif_dark_channel)
        
        elif self.normalize_method == 'local':
            # 局部归一化（每个batch内归一化）
            batch_size = dif_dark_channel.shape[0]
            normalized_dif = torch.zeros_like(dif_dark_channel)
            for i in range(batch_size):
                batch_dif = dif_dark_channel[i:i+1]
                batch_min = batch_dif.min()
                batch_max = batch_dif.max()
                if batch_max > batch_min:
                    normalized_dif[i:i+1] = (batch_dif - batch_min) / (batch_max - batch_min)
            dif_dark_channel = normalized_dif
        
        # 使用指数函数创建更平滑的权重分布
        # 这样可以让像差严重的区域获得更高的权重
        exp_dif = torch.exp(dif_dark_channel) - 1.0
        exp_dif = exp_dif / (torch.exp(torch.ones_like(dif_dark_channel)) - 1.0)  # 归一化到[0,1]
        
        # 创建AIF mask，范围在 [1, 1+mask_weight]
        AIF_mask = 1 + self.mask_weight * exp_dif
        
        # 对mask进行平滑处理
        if self.smooth_mask:
            # 使用平均池化进行平滑
            AIF_mask = F.avg_pool2d(AIF_mask, kernel_size=3, stride=1, padding=1)
        
        # 扩展到3通道
        AIF_mask = AIF_mask.repeat(1, 3, 1, 1)
        
        # 计算加权L1损失
        weighted_pred = AIF_mask * pred
        weighted_target = AIF_mask * target
        AIF_loss = self.loss_weight * l1_loss(weighted_pred, weighted_target, reduction='mean')
        
        return AIF_loss

@LOSS_REGISTRY.register()
class MultiScale_DIF_AIF_Loss(nn.Module):
    """多尺度DIF_AIF_Loss，在不同尺度上计算暗通道差异"""
    def __init__(self, loss_weight=1.0, mask_weight=2.0, scales=[1, 0.5, 0.25]):
        super(MultiScale_DIF_AIF_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.mask_weight = mask_weight
        self.scales = scales

    def forward(self, pred, target):
        total_loss = 0.0
        
        for scale in self.scales:
            # 缩放图像
            if scale != 1.0:
                scaled_pred = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                scaled_target = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                scaled_pred = pred
                scaled_target = target
            
            # 计算暗通道
            pred_dark = get_dark_channel(scaled_pred, 15)
            gt_dark = get_dark_channel(scaled_target, 15)
            
            # 计算差异
            dif_dark = torch.abs(pred_dark - gt_dark)
            
            # 归一化
            dif_min = dif_dark.min()
            dif_max = dif_dark.max()
            if dif_max > dif_min:
                dif_dark = (dif_dark - dif_min) / (dif_max - dif_min)
            
            # 创建mask
            AIF_mask = 1 + self.mask_weight * dif_dark
            AIF_mask = AIF_mask.repeat(1, 3, 1, 1)
            
            # 计算损失
            if scale != 1.0:
                # 将mask上采样回原始尺寸
                AIF_mask = F.interpolate(AIF_mask, size=pred.shape[2:], mode='bilinear', align_corners=False)
            
            weighted_pred = AIF_mask * pred
            weighted_target = AIF_mask * target
            scale_loss = l1_loss(weighted_pred, weighted_target, reduction='mean')
            
            total_loss += scale_loss * scale  # 给不同尺度不同权重
        
        return self.loss_weight * total_loss

@LOSS_REGISTRY.register()
class Universal_Aberration_Loss(nn.Module):
    """专门为universal像差恢复设计的loss，适用于各种严重程度的像差"""
    def __init__(self, loss_weight=1.0, adaptive_weight=True, severity_threshold=0.5):
        super(Universal_Aberration_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.adaptive_weight = adaptive_weight  # 是否使用自适应权重
        self.severity_threshold = severity_threshold  # 像差严重程度阈值

    def forward(self, pred, target):
        # 计算暗通道
        pred_dark = get_dark_channel(pred, 15)
        gt_dark = get_dark_channel(target, 15)
        
        # 计算暗通道差异
        dif_dark = torch.abs(pred_dark - gt_dark)
        
        # 自适应权重策略
        if self.adaptive_weight:
            # 计算每个batch的像差严重程度
            batch_severity = torch.mean(dif_dark, dim=[1,2,3])  # [B]
            
            # 根据严重程度动态调整权重
            # 轻度像差：权重1-2，重度像差：权重2-5
            adaptive_weights = torch.clamp(1.0 + 4.0 * batch_severity, 1.0, 5.0)
            
            # 将batch权重扩展到空间维度
            adaptive_weights = adaptive_weights.view(-1, 1, 1, 1)
            
            # 创建自适应mask
            # 轻度像差：更关注细节差异
            # 重度像差：更关注整体结构
            normalized_dif = dif_dark / (torch.mean(dif_dark, dim=[1,2,3], keepdim=True) + 1e-8)
            
            # 使用sigmoid函数创建平滑的权重分布
            sigmoid_weights = torch.sigmoid(10 * (normalized_dif - 1.0))
            
            # 结合自适应权重和局部差异
            AIF_mask = 1 + adaptive_weights * sigmoid_weights
            
        else:
            # 固定权重策略
            # 归一化差异到[0,1]
            dif_min = dif_dark.min()
            dif_max = dif_dark.max()
            if dif_max > dif_min:
                normalized_dif = (dif_dark - dif_min) / (dif_max - dif_min)
            else:
                normalized_dif = torch.zeros_like(dif_dark)
            
            # 使用平方函数增强重度像差区域的权重
            AIF_mask = 1 + 3.0 * (normalized_dif ** 2)
        
        # 扩展到3通道
        AIF_mask = AIF_mask.repeat(1, 3, 1, 1)
        
        # 计算加权L1损失
        weighted_pred = AIF_mask * pred
        weighted_target = AIF_mask * target
        loss = self.loss_weight * l1_loss(weighted_pred, weighted_target, reduction='mean')
        
        return loss

@LOSS_REGISTRY.register()
class Progressive_Aberration_Loss(nn.Module):
    """渐进式像差恢复loss，根据训练进度调整关注点"""
    def __init__(self, loss_weight=1.0, current_epoch=0, total_epochs=100):
        super(Progressive_Aberration_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    def forward(self, pred, target):
        # 计算暗通道差异
        pred_dark = get_dark_channel(pred, 15)
        gt_dark = get_dark_channel(target, 15)
        dif_dark = torch.abs(pred_dark - gt_dark)
        
        # 计算训练进度比例
        progress = self.current_epoch / self.total_epochs
        
        # 根据训练进度调整策略
        if progress < 0.3:
            # 早期训练：关注整体结构，权重相对均匀
            normalized_dif = dif_dark / (torch.mean(dif_dark) + 1e-8)
            mask_weight = 1.5
            AIF_mask = 1 + mask_weight * torch.clamp(normalized_dif, 0, 1)
            
        elif progress < 0.7:
            # 中期训练：开始关注像差区域
            dif_min = dif_dark.min()
            dif_max = dif_dark.max()
            if dif_max > dif_min:
                normalized_dif = (dif_dark - dif_min) / (dif_max - dif_min)
            else:
                normalized_dif = torch.zeros_like(dif_dark)
            
            mask_weight = 2.0 + progress * 2.0  # 权重逐渐增加
            AIF_mask = 1 + mask_weight * normalized_dif
            
        else:
            # 后期训练：重点关注重度像差区域
            dif_min = dif_dark.min()
            dif_max = dif_dark.max()
            if dif_max > dif_min:
                normalized_dif = (dif_dark - dif_min) / (dif_max - dif_min)
            else:
                normalized_dif = torch.zeros_like(dif_dark)
            
            # 使用指数函数增强重度像差区域
            exp_weights = torch.exp(2 * normalized_dif) - 1
            exp_weights = exp_weights / (torch.exp(torch.ones_like(normalized_dif) * 2) - 1)
            
            mask_weight = 3.0
            AIF_mask = 1 + mask_weight * exp_weights
        
        # 扩展到3通道
        AIF_mask = AIF_mask.repeat(1, 3, 1, 1)
        
        # 计算损失
        weighted_pred = AIF_mask * pred
        weighted_target = AIF_mask * target
        loss = self.loss_weight * l1_loss(weighted_pred, weighted_target, reduction='mean')
        
        return loss

    def update_epoch(self, epoch):
        """更新当前训练轮数"""
        self.current_epoch = epoch

# #主函数
# if __name__ == '__main__':
#     LOSS = fft_l1loss()
#     pred = torch.randn(1, 3, 256, 256)
#     target = torch.randn(1, 3, 256, 256)
#     loss = LOSS(pred, target)
#     print(loss)