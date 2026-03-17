import torch
import torch.nn as nn
from einops import rearrange


class FE(nn.Module):
    """Power-law feature extraction.

    Compresses STFT magnitude via power-law, preserving phase direction.
    Output has same real/imag structure but with compressed magnitudes.
    """

    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        """x: (B,F,T,2) -> (B,2,T,F)"""
        x_mag = torch.sqrt(x[..., [0]] ** 2 + x[..., [1]] ** 2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1 - self.c) + 1e-12)
        return x_c.permute(0, 3, 2, 1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])  # causal: pad top only
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4, 3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x):
        """x: (B,C,T,F)"""
        return self.elu(self.bn(self.conv(self.pad(x)))) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), stride=(1, 2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])  # causal
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)

    def forward(self, x):
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """x: (B,C,T,F) -> (B,C,T,F)"""
        y = rearrange(x, "b c t f -> b t (c f)")
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, "b t (c f) -> b c t f", c=x.shape[1])
        return y


class SubpixelConv2d(nn.Module):
    """Upsamples frequency dimension by 2x via sub-pixel shuffle."""

    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])  # causal
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.is_last = is_last
        if not is_last:
            self.bn = nn.BatchNorm2d(out_channels)
            self.elu = nn.ELU()

    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y
