import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 
        output = weight @ v
        output = output.transpose(1, 2) 
        output = output.reshape(input_shape) 
        output = self.out_proj(output) 
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        residue = x 
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residue
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128, channel_multipliers=[1, 2, 4, 4], num_res_blocks=2, downsample_factor=8, use_attention=True):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_upsample = int(math.log2(downsample_factor))
        
        channels = [base_channels * m for m in channel_multipliers]
        
        self.conv_in = nn.Conv2d(latent_channels, latent_channels, kernel_size=1, padding=0)
        self.conv_in2 = nn.Conv2d(latent_channels, channels[-1], kernel_size=3, padding=1)
        
        self.decoder_blocks = nn.ModuleList()
        
        for _ in range(num_res_blocks):
            self.decoder_blocks.append(VAE_ResidualBlock(channels[-1], channels[-1]))
        
        if use_attention:
            self.decoder_blocks.append(VAE_AttentionBlock(channels[-1]))
        
        for _ in range(num_res_blocks):
            self.decoder_blocks.append(VAE_ResidualBlock(channels[-1], channels[-1]))
        
        for i in range(self.num_upsample):
            self.decoder_blocks.append(nn.Upsample(scale_factor=2))
            
            in_ch = channels[-(i+1)]
            out_ch = channels[-(i+2)] if i < len(channels)-1 else channels[0]
            self.decoder_blocks.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1))
            
            for _ in range(num_res_blocks):
                if out_ch != in_ch:
                    self.decoder_blocks.append(VAE_ResidualBlock(in_ch, out_ch))
                    in_ch = out_ch
                else:
                    self.decoder_blocks.append(VAE_ResidualBlock(in_ch, in_ch))
        
        self.norm_out = nn.GroupNorm(32, channels[0])
        self.conv_out = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
    
        x /= 0.18215  
        
        x = self.conv_in(x)
        x = self.conv_in2(x)
        
        for module in self.decoder_blocks:
            x = module(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x

class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128, channel_multipliers=[1, 2, 4, 4], num_res_blocks=2, downsample_factor=8, use_attention=True):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_downsample = int(math.log2(downsample_factor))
        
        channels = [base_channels * m for m in channel_multipliers]
        
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        
        self.encoder_blocks = nn.ModuleList()
        
        for _ in range(num_res_blocks):
            self.encoder_blocks.append(VAE_ResidualBlock(channels[0], channels[0]))
        
        for i in range(self.num_downsample):
            in_ch = channels[i] if i < len(channels) else channels[-1]
            out_ch = channels[i+1] if i+1 < len(channels) else channels[-1]
            
            self.encoder_blocks.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=0))
            
            for j in range(num_res_blocks):
                if j == 0:
                    self.encoder_blocks.append(VAE_ResidualBlock(in_ch, out_ch))
                else:
                    self.encoder_blocks.append(VAE_ResidualBlock(out_ch, out_ch))
        
        for _ in range(num_res_blocks):
            self.encoder_blocks.append(VAE_ResidualBlock(channels[-1], channels[-1]))
        
        if use_attention:
            self.encoder_blocks.append(VAE_AttentionBlock(channels[-1]))
        
        self.encoder_blocks.append(VAE_ResidualBlock(channels[-1], channels[-1]))
        
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], latent_channels * 2, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=1, padding=0)

    def forward(self, x, noise):
        x = self.conv_in(x)
        
        for module in self.encoder_blocks:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        x = self.conv_out2(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215  # 移除原始SD的缩放因子
        
        return x

def calculate_encoder_output_shape(input_shape, latent_channels, downsample_factor):
    if len(input_shape) == 4:
        batch_size, _, height, width = input_shape
    elif len(input_shape) == 3:
        _, height, width = input_shape
        batch_size = None
    else:
        raise ValueError("input_shape should be 4D (batch_size, channels, height, width) or 3D (channels, height, width)")
    
    output_height = height // downsample_factor
    output_width = width // downsample_factor
    
    if batch_size is not None:
        return (batch_size, latent_channels, output_height, output_width)
    else:
        return (latent_channels, output_height, output_width)

# class VAE(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, latent_channels=4, base_channels=128, 
#                  channel_multipliers=[1, 2, 4, 4], num_res_blocks=2, downsample_factor=8, use_attention=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.latent_channels = latent_channels
#         self.downsample_factor = downsample_factor
        
#         self.encoder = VAE_Encoder(in_channels, latent_channels, base_channels, channel_multipliers, 
#                                  num_res_blocks, downsample_factor, use_attention)
#         self.decoder = VAE_Decoder(latent_channels, out_channels, base_channels, channel_multipliers, 
#                                  num_res_blocks, downsample_factor, use_attention)
    
#     def get_encoder_output_shape(self, input_shape):
#         return calculate_encoder_output_shape(input_shape, self.latent_channels, self.downsample_factor)
    
#     def encode(self, x, noise):
#         return self.encoder(x, noise)
    
#     def decode(self, x):
#         return self.decoder(x)
    
#     def forward(self, x, noise):
#         latent = self.encode(x, noise)
#         return self.decode(latent)
    

# vae_28 = VAE(in_channels=3, out_channels=3, latent_channels=2, 
#              base_channels=64, downsample_factor=4, 
#              channel_multipliers=[1, 2, 2], num_res_blocks=1)

