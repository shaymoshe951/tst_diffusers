import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A double convolution block with ReLU activation and batch normalization.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, do_final_gelu=False):
        super(DoubleConv, self).__init__()
        self.do_final_gelu = do_final_gelu
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.do_final_gelu:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)

class Down(nn.Module):
    """
    Downsampling block with a double convolution followed by max pooling.
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, in_channels, do_final_gelu=True),
            DoubleConv(in_channels, out_channels),
        )
        self.temb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.temb(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return  x + emb

class Up(nn.Module):
    """
    Upsampling block with a double convolution followed by transposed convolution.
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, do_final_gelu=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.temb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, x_skip, t):
        t_emb = self.temb(t)
        x = self.up(x)
        xcomb = torch.cat([x_skip, x], dim=1)
        xconv = self.conv(xcomb)
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return xconv + t_emb.view(xconv.shape[0], xconv.shape[1], 1, 1)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class UNetS(nn.Module):
    def __init__(self, image_shape_hwc, device='cuda'):
        super(UNetS, self).__init__()
        self.image_shape_hwc = image_shape_hwc
        self.device = device

        # Finding the next power of 2 from the image shape
        if image_shape_hwc[0] != image_shape_hwc[1] or image_shape_hwc[0] % 2 != 0 :
            raise ValueError("Image shape must be square and even")
        new_size = 2**(image_shape_hwc[0].bit_length())
        img_resample_kernel_size = new_size-image_shape_hwc[0] + 1
        self.align_to_power2 = nn.ConvTranspose2d(
                                in_channels=1,
                                out_channels=1,
                                kernel_size=img_resample_kernel_size,
                                stride=1,
                                padding=0
                            )
        self.time_dim = new_size*4

        # Assuming input is Bx1x32x32 (after up-scaling)
        ch1 = new_size
        self.inc = DoubleConv(1, ch1)
        self.down1 = Down(ch1, ch1 * 2, self.time_dim)
        self.sa1 = SelfAttention(ch1 * 2, new_size // 2)
        self.down2 = Down(ch1 * 2, ch1 * 2, self.time_dim)
        self.sa2 = SelfAttention(ch1 * 2, new_size // 4)

        self.bot1 = DoubleConv(ch1 * 2, ch1 * 4)
        self.bot2 = DoubleConv(ch1 * 4, ch1 * 2)

        self.up1 = Up(ch1 * 4, ch1, self.time_dim)
        self.sa4 = SelfAttention(ch1, new_size // 2)
        self.up2 = Up(ch1 * 2, ch1, self.time_dim)
        self.sa5 = SelfAttention(ch1, new_size)

        self.outc1 = nn.Conv2d(ch1, 1, kernel_size=1)
        self.outc2 = nn.Conv2d(1, 1, kernel_size=img_resample_kernel_size, stride=1, padding=0)

    def forward(self, x, t):
        t = t.squeeze_()
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        xp = self.align_to_power2(x)

        x1 = self.inc(xp)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        xb = self.bot1(x3)
        xb = self.bot2(xb)

        x = self.up1(xb, x2, t)
        x = self.sa4(x)
        x = self.up2(x, x1, t)
        x = self.sa5(x)

        x = self.outc1(x)
        self.out = self.outc2(x)
        return self.out

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

