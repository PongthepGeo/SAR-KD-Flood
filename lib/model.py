import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class MLP_256(nn.Module):
    def __init__(self, in_channels=3, width=64, mlp_hidden=64, dropout=0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.embed(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.drop(x)
        return self.head(x)


class UnetLight(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        c1 = self.enc(x)
        b = self.bottleneck(self.pool(c1))
        u = self.up(b)
        d = self.dec(u)
        return self.final(d)


class _MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden, tokens_mlp, channels_mlp):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, tokens_mlp),
            nn.GELU(),
            nn.Linear(tokens_mlp, num_tokens),
        )
        self.channel_norm = nn.LayerNorm(hidden)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden, channels_mlp),
            nn.GELU(),
            nn.Linear(channels_mlp, hidden),
        )

    def forward(self, x):
        y = self.token_norm(x.permute(0, 2, 1))
        y = y.permute(0, 2, 1)
        y = self.token_mlp(y)
        x = x + y
        z = self.channel_norm(x.permute(0, 2, 1))
        z = self.channel_mlp(z)
        z = z.permute(0, 2, 1)
        x = x + z
        return x


class PSPMixer(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, patch=16, hidden=48,
                 depth=3, tokens_mlp=None, channels_mlp=None, img_size=256):
        super().__init__()
        self.patch = patch
        self.hidden = hidden
        self.img_size = img_size
        tokens_mlp = tokens_mlp or hidden
        channels_mlp = channels_mlp or hidden * 2
        self.embed = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch)
        num_tokens = (img_size // patch) ** 2
        self.blocks = nn.ModuleList(
            [_MixerBlock(num_tokens, hidden, tokens_mlp, channels_mlp)
             for _ in range(depth)]
        )
        self.up = nn.Upsample(scale_factor=patch, mode="bilinear", align_corners=False)
        self.conv_out = nn.Conv2d(hidden, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embed(x)
        B, hidden, H, W = x.shape
        x = x.flatten(2)
        for blk in self.blocks:
            x = blk(x)
        x = x.view(B, hidden, H, W)
        x = self.up(x)
        return self.conv_out(x)
