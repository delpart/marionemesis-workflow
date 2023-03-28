from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import numpy as np
from .utils import load_data_int, load_data_float


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return nn.functional.gelu(x + self.net(x))
        else:
            return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Down, self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool(x)
        z = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + z


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels//2)
        )
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim, out_channels
            )
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        z = self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + z


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.net = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels,channels)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0]*size[1]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.net(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])


class MarioUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 256)
        self.bot2 = DoubleConv(256, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        device = next(self.parameters()).device
        inv_freq = 1.0 / (1000**(torch.arange(0, channels, 2, device=device)).float()/channels)
        pos_enc_a = torch.sin(t.repeat(1, channels//2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels//2)*inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


class MarioDiffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.2, level_size=(16, 32), device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.level_size = level_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_levels(self, x,t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        e = torch.rand_like(x)
        return sqrt_alpha_hat*x+sqrt_one_minus_alpha_hat*e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 7, *self.level_size[0])).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
        model.train()
        x = x.clamp(0, 1)*6
        return x

    def sample_continuation(self, model, level):
        model.eval()
        with torch.no_grad():
            x_orig = level.to(self.device).unsqueeze(0)
            x = torch.randn(x_orig.shape).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(1) * i).long().to(self.device)
                level_noise, _ = self.noise_levels(x_orig, t)
                x = torch.concat((level_noise[:,:,:-1,:], x[:,:,-1:,:]), dim=2)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise

        x = x.cpu().numpy()[:,:,-1,:].squeeze().T
        x = x.clip(0, 1)
        x = np.rint(x * 6)

        from .nemesis import int_to_char

        level_str = ''
        for i in range(x.shape[0]):
            level_str += int_to_char[min(6, x[i])]

        return level_str


def train(output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    training_tensors, _ = load_data_float()
    training_tensors = training_tensors
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = MarioUNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    diffusion = MarioDiffusion(device=device)

    epochs = trange(1000)
    losses = []
    for epoch in epochs:
        losses.append([])
        for i, levels in enumerate(data_loader):
            if len(levels) == 1:
                levels = levels[0]
            levels = levels[...,1:].to(device)
            t = diffusion.sample_timesteps(levels.shape[0]).to(device)
            x_t, noise = diffusion.noise_levels(levels, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses[-1].append(loss.item())

            epochs.set_description(f'Loss: {np.mean(losses[-1])}')

        if (epoch + 1)%20 == 0 or epoch == 999:
            torch.save(model.state_dict(), f'{output_dir}/{epoch}_chkpt.pt')




