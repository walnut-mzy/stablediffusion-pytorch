import torch
from torch import nn
from torch.nn import functional as F
import math
from VAE import *
from diffusion import *

class LatentClassifier(nn.Module):
    def __init__(self, latent_channels=2, latent_size=8, num_classes=10, hidden_dim=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.num_classes = num_classes
        
        input_dim = latent_channels * latent_size * latent_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, t=None):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        if t is not None:
            t_normalized = t.float().unsqueeze(-1) / 1000.0
            t_embed = self.time_embed(t_normalized)
            
            combined = torch.cat([x_flat, t_embed], dim=1)
            logits = self.fusion(combined)
        else:
            logits = self.classifier(x_flat)
        
        return logits

class stablediffusion(nn.Module):
    def __init__(self, 
                in_channels=3, 
                 out_channels=3,
                 latent_channels=4, 
                 base_channels=128, 
                 channel_multipliers=[1, 2, 4, 4], 
                 num_res_blocks=2, 
                 downsample_factor=8, 
                 use_attention=True,
                 unet_base_channels=64,
                 attention_resolutions=(4, 8),
                 dropout=0.1,
                 channel_mult=(1, 2, 4),
                 time_emb_dim=256,
                 num_classes=None,
                 num_timesteps=1000,
                 input_image_size=32,
                 use_classifier_guidance=False,
                 classifier_hidden_dim=256,
                 vae_weights_path=None,  # VAE weights path parameter
                 freeze_vae=False  # Add parameter to control VAE freezing
                 ):
        super().__init__()
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.input_image_size = input_image_size
        self.out_channels = out_channels
        self.use_classifier_guidance = use_classifier_guidance
        self.num_classes = num_classes
        
        self.encoder = VAE_Encoder(in_channels, latent_channels, base_channels, channel_multipliers, 
                                 num_res_blocks, downsample_factor, use_attention)
        
        self.decoder = VAE_Decoder(latent_channels, out_channels, base_channels, channel_multipliers, 
                                 num_res_blocks, downsample_factor, use_attention)
        
        # Load and optionally freeze VAE weights
        self._load_and_freeze_vae(vae_weights_path, freeze_vae)
        
        batch_size, latent_ch, latent_h, latent_w = self.get_encoder_output_shape((1, in_channels, input_image_size, input_image_size))
        
        print(f"Latent空间形状: ({batch_size}, {latent_ch}, {latent_h}, {latent_w})")
        
        self.unet = create_unet(
            image_size=latent_h,
            in_channels=latent_ch,
            base_channels=unet_base_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            time_emb_dim=time_emb_dim,
            time_emb_scale=1.0,
            num_classes=num_classes,
            norm="gn",
            num_groups=8,
            activation=F.relu,
            initial_pad=0,
        )

        self.diffusion_model = DiffusionModel(
            model=self.unet,
            num_steps=num_timesteps,
            img_channels=latent_ch,
            img_size=(latent_h, latent_w),
            ema_decay=0.999,
            ema_start=1000,
            ema_update_rate=1,
        )
        
        if use_classifier_guidance and num_classes is not None:
            self.classifier = LatentClassifier(
                latent_channels=latent_ch,
                latent_size=latent_h,
                num_classes=num_classes,
                hidden_dim=classifier_hidden_dim
            )
        else:
            self.classifier = None
        
    def _load_and_freeze_vae(self, weights_path, freeze_vae):
        """
        Load VAE weights and optionally freeze the parameters
        
        Args:
            weights_path: Path to the VAE weights file
            freeze_vae: Whether to freeze VAE parameters
        """
        if weights_path is None:
            if freeze_vae:
                print("Warning: freeze_vae=True but no weights_path provided. VAE parameters will NOT be frozen.")
            return
            
        try:
            # Load the weights
            state_dict = torch.load(weights_path)
            
            # Load encoder weights
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            if encoder_state_dict:
                self.encoder.load_state_dict(encoder_state_dict)
            
            # Load decoder weights
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
            if decoder_state_dict:
                self.decoder.load_state_dict(decoder_state_dict)
            
            # Optionally freeze parameters
            if freeze_vae:
                self.freeze_vae_parameters()
                print(f"Successfully loaded and froze VAE weights from {weights_path}")
            else:
                print(f"Successfully loaded VAE weights from {weights_path} (parameters are trainable)")
            
        except Exception as e:
            print(f"Warning: Failed to load VAE weights from {weights_path}: {str(e)}")
            print("Continuing with randomly initialized VAE weights")
            
    def freeze_vae_parameters(self):
        """
        Freeze VAE encoder and decoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        print("VAE parameters have been frozen")
    
    def unfreeze_vae_parameters(self):
        """
        Unfreeze VAE encoder and decoder parameters
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        print("VAE parameters have been unfrozen")
    
    def load_vae_weights(self, weights_path, freeze_after_load=False):
        """
        Load VAE weights from file
        
        Args:
            weights_path: Path to the VAE weights file
            freeze_after_load: Whether to freeze VAE parameters after loading
        """
        try:
            state_dict = torch.load(weights_path)
            
            # Load encoder weights
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            if encoder_state_dict:
                self.encoder.load_state_dict(encoder_state_dict)
                print("Encoder weights loaded successfully")
            
            # Load decoder weights  
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
            if decoder_state_dict:
                self.decoder.load_state_dict(decoder_state_dict)
                print("Decoder weights loaded successfully")
            
            if freeze_after_load:
                self.freeze_vae_parameters()
                
        except Exception as e:
            print(f"Error loading VAE weights from {weights_path}: {str(e)}")
            raise e
    
    def save_vae_weights(self, save_path):
        """
        Save only VAE weights to file
        
        Args:
            save_path: Path to save VAE weights
        """
        vae_state_dict = {}
        
        # Save encoder weights
        for name, param in self.encoder.named_parameters():
            vae_state_dict[f'encoder.{name}'] = param.data
        
        # Save decoder weights
        for name, param in self.decoder.named_parameters():
            vae_state_dict[f'decoder.{name}'] = param.data
        
        torch.save(vae_state_dict, save_path)
        print(f"VAE weights saved to: {save_path}")
    
    def get_encoder_output_shape(self, input_shape):
        return calculate_encoder_output_shape(input_shape, self.latent_channels, self.downsample_factor)
    
    def sample_latent(self, batch_size, device, y=None, use_ema=True):
        return self.diffusion_model.sample_diffusion_sequence(
            batch_size=batch_size, 
            device=device, 
            y=y, 
            use_ema=use_ema
        )
    
    def sample_with_classifier_guidance(self, batch_size, device, target_class, guidance_scale=1.0, use_ema=True):
        if self.classifier is None:
            raise ValueError("Classifier guidance is not enabled. Set use_classifier_guidance=True")
        
        y = torch.full((batch_size,), target_class, device=device, dtype=torch.long)
        
        latent_shape = self.get_encoder_output_shape((batch_size, 3, self.input_image_size, self.input_image_size))
        x = torch.randn(latent_shape, device=device)
        
        diffusion_sequence = [x.cpu().detach()]
        
        unet_model = self.diffusion_model.ema_model if use_ema else self.diffusion_model.model
        
        for t in range(self.diffusion_model.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            
            with torch.no_grad():
                # 生成无条件噪声预测（传递y=None）
                noise_pred_uncond = unet_model(x, time=t_batch, y=None)
                
                # 生成条件噪声预测
                if self.num_classes is not None:
                    noise_pred_cond = unet_model(x, time=t_batch, y=y)
                else:
                    noise_pred_cond = noise_pred_uncond
            
            if guidance_scale > 0:
                # 为了计算梯度，需要在有梯度的上下文中预测x0
                with torch.enable_grad():
                    # 将x设置为需要梯度的tensor
                    x_for_grad = x.detach().requires_grad_(True)
                    x_0_pred = self._predict_x0_from_noise(x_for_grad, noise_pred_uncond.detach(), t_batch)
                    
                    classifier_logits = self.classifier(x_0_pred, t_batch)
                    
                    classifier_guidance = torch.autograd.grad(
                        outputs=F.log_softmax(classifier_logits, dim=-1)[range(batch_size), y].sum(),
                        inputs=x_for_grad,
                        create_graph=False
                    )[0]
                
                noise_pred = noise_pred_uncond + guidance_scale * classifier_guidance
            else:
                noise_pred = noise_pred_uncond
            
            x = self._denoise_step(x, noise_pred, t_batch)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.diffusion_model.extract(self.diffusion_model.sigma, t_batch, x.shape)
                x += sigma * noise
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence
    
    def _predict_x0_from_noise(self, x_t, noise_pred, t):
        sqrt_alphas_cumprod = self.diffusion_model.extract(self.diffusion_model.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod = self.diffusion_model.extract(self.diffusion_model.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod * noise_pred) / sqrt_alphas_cumprod
        return x_0_pred
    
    def _denoise_step(self, x_t, noise_pred, t):
        reciprocal_sqrt_alphas = self.diffusion_model.extract(self.diffusion_model.reciprocal_sqrt_alphas, t, x_t.shape)
        remove_noise_coeff = self.diffusion_model.extract(self.diffusion_model.remove_noise_coeff, t, x_t.shape)
        
        x_prev = reciprocal_sqrt_alphas * (x_t - remove_noise_coeff * noise_pred)
        return x_prev
    
    def train_classifier(self, latent, labels, t=None):
        if self.classifier is None:
            raise ValueError("Classifier is not initialized")
        
        logits = self.classifier(latent, t)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, x, noise=None, mode='reconstruction', labels=None):
        if mode == 'reconstruction':
            if noise is None:
                latent_shape = self.get_encoder_output_shape(x.shape)
                noise = torch.randn(latent_shape, device=x.device)
            
            latent = self.encoder(x, noise)
            reconstructed = self.decoder(latent)
            return reconstructed
            
        elif mode == 'diffusion':
            if noise is None:
                latent_shape = self.get_encoder_output_shape(x.shape)
                noise = torch.randn(latent_shape, device=x.device)
            
            latent = self.encoder(x, noise)
            loss = self.diffusion_model(latent, labels)
            return loss
            
        elif mode == 'classifier':
            if self.classifier is None:
                raise ValueError("Classifier is not initialized")
            if labels is None:
                raise ValueError("Labels are required for classifier training")
            
            if noise is None:
                latent_shape = self.get_encoder_output_shape(x.shape)
                noise = torch.randn(latent_shape, device=x.device)
            
            with torch.no_grad():
                latent = self.encoder(x, noise)
            
            batch_size = latent.shape[0]
            t = torch.randint(0, self.diffusion_model.num_timesteps, (batch_size,), device=x.device)
            
            noise_latent = torch.randn_like(latent)
            sqrt_alphas_cumprod = self.diffusion_model.extract(self.diffusion_model.sqrt_alphas_cumprod, t, latent.shape)
            sqrt_one_minus_alphas_cumprod = self.diffusion_model.extract(self.diffusion_model.sqrt_one_minus_alphas_cumprod, t, latent.shape)
            
            noisy_latent = sqrt_alphas_cumprod * latent + sqrt_one_minus_alphas_cumprod * noise_latent
            
            classifier_loss = self.train_classifier(noisy_latent, labels, t)
            return classifier_loss
        else:
            raise ValueError("mode must be 'reconstruction', 'diffusion', or 'classifier'")


def stable_diffusion_for_minist(use_classifier_guidance=True):
    model = stablediffusion(
        # 图像参数
        in_channels=3,
        out_channels=3,
        input_image_size=32,
        
        # VAE编码器参数 - 增加容量但保持轻量
        latent_channels=4,  
        base_channels=64,           
        channel_multipliers=[1, 2, 4], 
        num_res_blocks=2,           
        downsample_factor=8,        
        use_attention=True,
        
        # UNet参数 - 与VAE保持一致
        unet_base_channels=32,     
        attention_resolutions=(2, 4),  
        dropout=0.1,
        channel_mult=(1, 2, 4),     
        time_emb_dim=128,        
        
        # 分类参数
        num_classes=10 if use_classifier_guidance else None,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=256,
        
        # 扩散参数
        num_timesteps=1000,
        
        # VAE控制参数
        vae_weights_path=None,  # 不自动加载VAE权重
        freeze_vae=False        # 不自动冻结VAE参数
    )
    return model

def stable_diffusion_for_cifar10(use_classifier_guidance=True):
    """
    为CIFAR-10优化的Stable Diffusion模型
    - 输入图像尺寸: 32x32x3
    - 潜在空间: 8x8x4
    - UNet和VAE使用更深的网络结构
    """
    return stablediffusion(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        base_channels=128,          # VAE基础通道数
        channel_multipliers=[1, 2, 4], # VAE通道倍数
        num_res_blocks=2,           # VAE残差块数
        downsample_factor=4,        # VAE下采样倍数 (32 / 4 = 8)
        use_attention=True,
        unet_base_channels=128,      # UNet基础通道数
        attention_resolutions=(4,), # 在4x4的特征图上使用注意力
        dropout=0.1,
        channel_mult=(1, 2, 2, 4),  # UNet通道倍数
        time_emb_dim=256,
        num_classes=10 if use_classifier_guidance else None,
        num_timesteps=1000,
        input_image_size=32,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=256,
        vae_weights_path=None,      # 不自动加载VAE权重
        freeze_vae=False            # 不自动冻结VAE参数
    )

def stable_diffusion_for_imagenet_64(use_classifier_guidance=True, num_classes=1000):
    """
    为ImageNet 64x64设计的Stable Diffusion模型
    """
    model = stablediffusion(
        # 图像参数
        in_channels=3,
        out_channels=3,
        input_image_size=64,
        
        # VAE编码器参数 - 适合64x64图像
        latent_channels=8,              # 增加潜在空间通道数
        base_channels=128,              # 增加基础通道数
        channel_multipliers=[1, 2, 4, 8],  # 更深的网络结构
        num_res_blocks=2,               # 保持2个残差块
        downsample_factor=8,            # 8倍下采样，64->8
        use_attention=True,
        
        # UNet参数 - 匹配潜在空间尺寸(8x8)
        unet_base_channels=128,         # 增加UNet基础通道数
        attention_resolutions=(2, 4),   # 适合8x8的潜在空间
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),      # 与VAE保持一致
        time_emb_dim=512,               # 增加时间嵌入维度
        
        # 分类参数
        num_classes=num_classes,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=512,      # 增加分类器隐藏维度
        
        # 扩散参数
        num_timesteps=1000,
        
        # VAE控制参数
        vae_weights_path=None,          # 不自动加载VAE权重
        freeze_vae=False                # 不自动冻结VAE参数
    )
    return model

def stable_diffusion_for_imagenet_128(use_classifier_guidance=True, num_classes=1000):
    """
    为ImageNet 128x128设计的Stable Diffusion模型
    """
    model = stablediffusion(
        # 图像参数
        in_channels=3,
        out_channels=3,
        input_image_size=128,
        
        # VAE编码器参数 - 适合128x128图像
        latent_channels=8,              # 潜在空间通道数
        base_channels=128,              # 基础通道数
        channel_multipliers=[1, 2, 4, 8],  # 深层网络结构
        num_res_blocks=2,               # 残差块数量
        downsample_factor=8,            # 8倍下采样，128->16
        use_attention=True,
        
        # UNet参数 - 匹配潜在空间尺寸(16x16)
        unet_base_channels=128,         # UNet基础通道数
        attention_resolutions=(4, 8),   # 适合16x16的潜在空间
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),      # 与VAE保持一致
        time_emb_dim=512,               # 时间嵌入维度
        
        # 分类参数
        num_classes=num_classes,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=512,
        
        # 扩散参数
        num_timesteps=1000,
        
        # VAE控制参数
        vae_weights_path=None,          # 不自动加载VAE权重
        freeze_vae=False                # 不自动冻结VAE参数
    )
    return model

def stable_diffusion_for_imagenet_256(use_classifier_guidance=True, num_classes=1000):
    """
    为ImageNet 256x256设计的Stable Diffusion模型
    高质量但计算开销大
    """
    model = stablediffusion(
        # 图像参数
        in_channels=3,
        out_channels=3,
        input_image_size=256,
        
        # VAE编码器参数 - 适合256x256图像
        latent_channels=8,              # 潜在空间通道数
        base_channels=128,              # 基础通道数
        channel_multipliers=[1, 2, 4, 8],  # 深层网络结构
        num_res_blocks=2,               # 残差块数量
        downsample_factor=8,            # 8倍下采样，256->32
        use_attention=True,
        
        # UNet参数 - 匹配潜在空间尺寸(32x32)
        unet_base_channels=128,         # UNet基础通道数
        attention_resolutions=(8, 16),  # 适合32x32的潜在空间
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),      # 与VAE保持一致
        time_emb_dim=512,               # 时间嵌入维度
        
        # 分类参数
        num_classes=num_classes,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=512,
        
        # 扩散参数
        num_timesteps=1000,
        
        # VAE控制参数
        vae_weights_path=None,          # 不自动加载VAE权重
        freeze_vae=False                # 不自动冻结VAE参数
    )
    return model

def stable_diffusion_for_imagenet_small(use_classifier_guidance=True, num_classes=100):
    """
    为ImageNet子集设计的轻量级模型
    适合快速实验和资源受限的环境
    """
    model = stablediffusion(
        # 图像参数
        in_channels=3,
        out_channels=3,
        input_image_size=64,
        
        # VAE编码器参数 - 轻量级设计
        latent_channels=4,              # 较小的潜在空间
        base_channels=64,               # 较小的基础通道数
        channel_multipliers=[1, 2, 4],  # 较浅的网络
        num_res_blocks=1,               # 少量残差块
        downsample_factor=8,            # 8倍下采样
        use_attention=True,
        
        # UNet参数
        unet_base_channels=64,          # 较小的UNet
        attention_resolutions=(2, 4),   # 适合小潜在空间
        dropout=0.1,
        channel_mult=(1, 2, 4),         # 与VAE保持一致
        time_emb_dim=256,               # 较小的时间嵌入
        
        # 分类参数
        num_classes=num_classes,
        use_classifier_guidance=use_classifier_guidance,
        classifier_hidden_dim=256,      # 较小的分类器
        
        # 扩散参数
        num_timesteps=1000,
        
        # VAE控制参数
        vae_weights_path=None,          # 不自动加载VAE权重
        freeze_vae=False                # 不自动冻结VAE参数
    )
    return model

def get_model_info(model):
    """
    获取模型信息和参数统计
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分别统计各组件参数
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    unet_params = sum(p.numel() for p in model.unet.parameters())
    diffusion_params = sum(p.numel() for p in model.diffusion_model.parameters())
    
    classifier_params = 0
    if model.classifier is not None:
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    # 计算潜在空间尺寸
    latent_shape = model.get_encoder_output_shape((1, 3, model.input_image_size, model.input_image_size))
    
    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "unet_params": unet_params,
        "diffusion_params": diffusion_params,
        "classifier_params": classifier_params,
        "input_image_size": model.input_image_size,
        "latent_shape": latent_shape,
        "latent_channels": model.latent_channels,
        "num_classes": model.num_classes,
        "use_classifier_guidance": model.use_classifier_guidance,
        "num_timesteps": model.diffusion_model.num_timesteps
    }
    
    return info

def print_model_info(model, model_name="Stable Diffusion"):
    """
    打印模型详细信息
    """
    info = get_model_info(model)
    
    print(f"\n{'='*60}")
    print(f"{model_name} 模型信息")
    print('='*60)
    
    print(f"\n[模型结构]")
    print(f"输入图像尺寸: {info['input_image_size']}x{info['input_image_size']}")
    print(f"潜在空间形状: {info['latent_shape']}")
    print(f"潜在空间通道数: {info['latent_channels']}")
    print(f"类别数量: {info['num_classes']}")
    print(f"扩散步数: {info['num_timesteps']}")
    print(f"使用分类器引导: {info['use_classifier_guidance']}")
    
    print(f"\n[参数统计]")
    print(f"总参数量: {info['total_params']:,}")
    print(f"可训练参数: {info['trainable_params']:,}")
    print(f"  - VAE编码器: {info['encoder_params']:,}")
    print(f"  - VAE解码器: {info['decoder_params']:,}")
    print(f"  - UNet: {info['unet_params']:,}")
    print(f"  - 扩散模型: {info['diffusion_params']:,}")
    if info['classifier_params'] > 0:
        print(f"  - 分类器: {info['classifier_params']:,}")
    
    # 估算显存使用量（粗略估计）
    latent_size = info['latent_shape'][1] * info['latent_shape'][2] * info['latent_shape'][3]
    memory_est_mb = (info['total_params'] * 4 + latent_size * 4 * 32) / (1024 * 1024)  # 假设32的batch size
    print(f"\n[显存估算]")
    print(f"估算显存使用量: ~{memory_est_mb:.1f} MB (batch_size=32)")
    
    print('='*60 + "\n")

# 预定义的配置选择函数
def get_imagenet_model(image_size=64, num_classes=1000, use_classifier_guidance=True, model_size="normal"):
    """
    根据参数选择合适的ImageNet模型
    
    Args:
        image_size: 图像尺寸 (64, 128, 256)
        num_classes: 类别数量 (1000为完整ImageNet, 可设置更小值用于子集)
        use_classifier_guidance: 是否使用分类器引导
        model_size: 模型大小 ("small", "normal", "large")
    """
    
    if model_size == "small" or num_classes <= 100:
        model = stable_diffusion_for_imagenet_small(
            use_classifier_guidance=use_classifier_guidance,
            num_classes=num_classes
        )
        model_name = f"Stable Diffusion ImageNet Small ({image_size}x{image_size}, {num_classes} classes)"
    
    elif image_size == 64:
        model = stable_diffusion_for_imagenet_64(
            use_classifier_guidance=use_classifier_guidance,
            num_classes=num_classes
        )
        model_name = f"Stable Diffusion ImageNet 64x64 ({num_classes} classes)"
    
    elif image_size == 128:
        model = stable_diffusion_for_imagenet_128(
            use_classifier_guidance=use_classifier_guidance,
            num_classes=num_classes
        )
        model_name = f"Stable Diffusion ImageNet 128x128 ({num_classes} classes)"
    
    elif image_size == 256:
        model = stable_diffusion_for_imagenet_256(
            use_classifier_guidance=use_classifier_guidance,
            num_classes=num_classes
        )
        model_name = f"Stable Diffusion ImageNet 256x256 ({num_classes} classes)"
    
    else:
        raise ValueError(f"不支持的图像尺寸: {image_size}. 支持的尺寸: 64, 128, 256")
    
    return model, model_name 