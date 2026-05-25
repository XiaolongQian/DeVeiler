import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math
from einops import repeat

"""
LOTGMP is the latent-map predictor used by VeilGen.
scale_vector = latent transmission map.
shift_vector = latent glare map.
"""

"""
改进的Latent Parameter Estimator (LOTGMP)
- 移除zt输入，避免循环依赖
- 只使用z_blur和t作为输入，更符合物理直觉
"""

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class SelfAttention(nn.Module):
    """带残差和LayerNorm的Self-Attention模块"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, f"channels {channels} must be divisible by num_heads {num_heads}"
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        
        # LayerNorm
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # 生成Q, K, V
        qkv = self.qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(B, self.num_heads, self.head_dim, N), qkv)
        
        # 调整形状
        q = q.transpose(-2, -1)  # (B, heads, N, dim)
        k = k.transpose(-2, -1)  # (B, heads, N, dim)
        v = v.transpose(-2, -1)  # (B, heads, N, dim)
        
        # 计算attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        
        # 应用attention
        out = (attn @ v)  # (B, heads, N, dim)
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        
        # 恢复形状
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # 投影 + 残差
        out = self.proj(out) + shortcut
        return out

class ResBlock(nn.Module):
    """残差块 - 参考ControlNet的ResBlock设计，支持时间编码调制"""
    
    def __init__(self, channels: int, emb_channels: int, use_attention: bool = False, use_scale_shift_norm: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 输入层 - 参考ControlNet的in_layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )
        
        # 时间编码层 - 参考ControlNet的emb_layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels if use_scale_shift_norm else channels)
        )
        
        # 输出层 - 参考ControlNet的out_layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )
        
        if use_attention:
            self.attention = SelfAttention(channels)
        
        # 跳跃连接
        self.skip_connection = nn.Identity()
        
    def forward(self, x, emb=None):
        """
        前向传播 - 支持时间编码调制，参考ControlNet的ResBlock
        
        Args:
            x: 输入特征 [B, C, H, W]
            emb: 时间编码 [B, emb_channels] (可选)
        """
        residual = x
        
        # 输入层处理
        h = self.in_layers(x)
        
        # 时间编码调制 - 参考ControlNet的调制方式
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            
            if self.use_scale_shift_norm:
                # 使用scale-shift调制 (FiLM-like) - 参考ControlNet
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                # 简单相加 - 参考ControlNet
                h = h + emb_out
                h = self.out_layers(h)
        else:
            # 没有时间编码时，正常处理
            h = self.out_layers(h)
        
        # 注意力机制
        if self.use_attention:
            h = self.attention(h)
        
        # 残差连接
        return self.skip_connection(residual) + h


class LOTGMP(nn.Module):
    """
    改进的Latent Parameter Estimator (LOTGMP)
    输入: x_noisy, z_blur, t (包含当前噪声状态)
    输出: scale_vector, shift_vector (与zt同shape)
    
    设计理念:
    - LOTGMP从退化图像和当前噪声状态学习物理参数
    - 结合当前噪声状态，提供更精确的调制参数
    - 训练和推理逻辑一致
    """
    
    def __init__(self, latent_channels: int = 4, hidden_channels: int = 128, 
                 num_res_blocks: int = 4, use_attention: bool = True, shift_scale: float = 3):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        
        # 时间编码维度 - 使用ControlNet标准设计
        time_embed_dim = hidden_channels * 4  # 512维 (128 * 4)
        
        # 时间embedding - 使用ControlNet标准设计
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 输入投影层 - 处理x_noisy和z_blur的拼接
        self.input_proj = nn.Sequential(
            nn.Conv2d(latent_channels * 2, hidden_channels, 3, 1, 1, bias=False),  # 8通道输入
            nn.GroupNorm(32, hidden_channels),
            nn.SiLU()
        )
        
        # 残差块序列 - 使用改进的ResBlock，支持时间编码调制
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            use_attn = use_attention and (i == num_res_blocks // 2)  # 在中间层使用attention
            self.res_blocks.append(ResBlock(
                channels=hidden_channels, 
                emb_channels=time_embed_dim,  # 传入时间编码维度
                use_attention=use_attn,
                use_scale_shift_norm=True  # 使用scale-shift调制
            ))
        
        # 输出投影层 - 分别预测scale和shift
        self.scale_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1, bias=False),
            nn.GroupNorm(32, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, latent_channels, 1, bias=False),
            nn.Tanh()  # scale范围[-1, 1]
        )
        
        # self.shift_proj = nn.Sequential(
        #     nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1, bias=False),
        #     nn.GroupNorm(32, hidden_channels // 2),
        #     nn.SiLU(),
        #     nn.Conv2d(hidden_channels // 2, latent_channels, 1, bias=False),
        #     nn.Sigmoid()  # shift范围[0, 1]
        # )

        self.shift_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1, bias=False),
            nn.GroupNorm(32, hidden_channels // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, latent_channels, 1, bias=False),
            nn.Tanh()  # shift范围[0, 1]
        )

        self.shift_scale = shift_scale
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        获取时间编码 - 使用ControlNet标准方式
        
        Args:
            t: 时间步 [B]
            
        Returns:
            [B, time_embed_dim] - 时间编码
        """
        # 使用ControlNet标准的时间embedding
        t_emb = timestep_embedding(t, self.hidden_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        return emb
    
    def forward(self, x_noisy: torch.Tensor, z_blur: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 参考ControlNet的时间编码调制方式
        
        Args:
            x_noisy: 当前噪声latent [B, 4, H, W]
            z_blur: 模糊图像latent [B, 4, H, W]
            t: 时间步 [B]
            
        Returns:
            Dict containing:
                - scale_vector: [B, 4, H, W]
                - shift_vector: [B, 4, H, W]
        """
        B, C, H, W = x_noisy.shape
        
        # 获取时间编码 - 参考ControlNet的方式
        emb = self._get_time_embedding(t)  # [B, time_embed_dim]
        
        # 拼接x_noisy和z_blur作为输入
        x_input = torch.cat([x_noisy, z_blur], dim=1)  # [B, 8, H, W]
        
        # 输入投影 - 处理拼接后的输入
        x = self.input_proj(x_input)  # [B, hidden_channels, H, W]
        
        # 残差块处理 - 每个ResBlock都会接收时间编码进行调制
        for res_block in self.res_blocks:
            x = res_block(x, emb)  # 传入时间编码
        
        # 预测scale和shift向量
        scale_vector = 0.99 * self.scale_proj(x)  # [B, 4, H, W]
        shift_vector = self.shift_scale*self.shift_proj(x)  # [B, 4, H, W]
        
        return {
            'scale_vector': scale_vector,
            'shift_vector': shift_vector
        }
    
    def modulate_latent(self, zt: torch.Tensor, scale_vector: torch.Tensor, 
                       shift_vector: torch.Tensor) -> torch.Tensor:
        """
        使用预测的参数调制latent
        
        Args:
            zt: 原始latent [B, 4, H, W]
            scale_vector: scale向量 [B, 4, H, W]
            shift_vector: shift向量 [B, 4, H, W]
            
        Returns:
            调制后的latent [B, 4, H, W]
        """
        # 打印阈值用于调试
        print(f"shift_vector range: [{shift_vector.min():.4f}, {shift_vector.max():.4f}]")
        print(f"scale_vector range: [{scale_vector.min():.4f}, {scale_vector.max():.4f}]")
        print(f"zt range: [{zt.min():.4f}, {zt.max():.4f}]")
        
        # 应用scale和shift调制
        modulated_zt = zt * (1 + scale_vector) + shift_vector
        return modulated_zt


def create_lotgmp(latent_channels: int = 4, hidden_channels: int = 128, 
                       num_res_blocks: int = 4, use_attention: bool = True, shift_scale: float = 3) -> LOTGMP:
    """创建改进版LOTGMP实例"""
    return LOTGMP(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention,
        shift_scale=shift_scale
    )


ImprovedLatentParameterEstimator = LOTGMP
create_improved_lotgmp = create_lotgmp
create_improved_lpe = create_lotgmp


# 测试代码
if __name__ == "__main__":
    # 创建改进版LOTGMP
    lotgmp = create_lotgmp(latent_channels=4, hidden_channels=128, num_res_blocks=4, use_attention=True, shift_scale=3)
    
    # 测试前向传播
    batch_size = 16
    latent_channels = 4
    height, width = 64, 64
    
    x_noisy = torch.randn(batch_size, latent_channels, height, width)
    z_blur = torch.randn(batch_size, latent_channels, height, width)
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shapes:")
    print(f"  x_noisy: {x_noisy.shape}")
    print(f"  z_blur: {z_blur.shape}")
    print(f"  t: {t.shape}")
    
    # 前向传播
    results = lotgmp(x_noisy, z_blur, t)
    scale_vector = results['scale_vector']
    shift_vector = results['shift_vector']
    
    print(f"\nOutput shapes:")
    print(f"  scale_vector: {scale_vector.shape}")
    print(f"  shift_vector: {shift_vector.shape}")
    
    # 测试调制（使用随机zt）
    zt = torch.randn(batch_size, latent_channels, height, width)
    modulated_zt = lotgmp.modulate_latent(zt, scale_vector, shift_vector)
    print(f"  modulated_zt: {modulated_zt.shape}")
    
    # 测试concat
    concat_zt = torch.cat([zt, modulated_zt], dim=1)
    print(f"  concat_zt: {concat_zt.shape}")
    
    print("\nImproved LOTGMP test completed successfully!")
