import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import numpy as np
from functools import partial
import math

from unet import *

def create_unet(
    image_size=32,
    in_channels=3,
    base_channels=128,
    out_channels=None,
    num_res_blocks=2,
    attention_resolutions=(),
    dropout=0.0,
    channel_mult=(1, 2, 4, 8),
    time_emb_dim=512,
    time_emb_scale=1.0,
    num_classes=None,
    norm="gn",
    num_groups=32,
    activation=F.relu,
    initial_pad=0,
):
    if out_channels is None:
        out_channels = in_channels
        
    return UNet(
        img_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mult,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        time_emb_scale=time_emb_scale,
        num_classes=num_classes,
        activation=activation,
        dropout=dropout,
        attention_resolutions=attention_resolutions,
        norm=norm,
        num_groups=num_groups,
        initial_pad=initial_pad,
    )

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class DiffusionModel(nn.Module):
    def __init__(
            self,
            model,
            loss=None,
            ema_decay=0.9999,
            ema_start=5000,
            ema_update_rate=1,
            num_steps=1000,
            schedule_s=0.008,
            img_channels=3,
            img_size=(32, 32),
                 ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.ema = EMA(self.ema_decay)
        self.ema_model = deepcopy(model)
        self.step = 0

        self.steps=num_steps
        self.num_timesteps = num_steps
        self.s = schedule_s
        self.img_channels = img_channels
        self.img_size = img_size

        betas = self.generate_cosine_schedule()
        self.register_buffer("betas", betas)

        self.loss=loss

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    def perturb_x(self, x, t, noise):
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        ) 
    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, time=t, y=y)

        if self.loss==None:
            loss = F.mse_loss(estimated_noise, noise)
        else:
            loss = self.loss(estimated_noise, noise)
        return loss
    def f(self, t, T):
        return (torch.cos((t / T + self.s) / (1 + self.s) * torch.pi / 2)) ** 2
    def generate_cosine_schedule(self):
        T = self.steps
        s = self.s
        timesteps = torch.arange(T + 1, dtype=torch.float64)
        alphas_cumprod = self.f(timesteps, T) / self.f(torch.tensor(0), T)
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    def generate_linear_schedule(self,T, low, high):
        return torch.linspace(low, high, T)
    
    @torch.no_grad()
    def recover_noise(self, x, t, y, use_ema=True):
        if use_ema:
            predicted_noise = self.ema_model(x, time=t, y=y)
            return (
                (x - self.extract(self.remove_noise_coeff, t, x.shape) * predicted_noise) *
                self.extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            predicted_noise = self.model(x, time=t, y=y)
            return (
                (x - self.extract(self.remove_noise_coeff, t, x.shape) * predicted_noise) *
                self.extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.recover_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += self.extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence
    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)
    
