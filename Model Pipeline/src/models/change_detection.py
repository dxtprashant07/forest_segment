import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Standard U-Net double convolution block"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for bi-temporal change detection

    Architecture: base_channels=32 (~10M params)
    Shared encoder for T1 and T2, concatenated bottleneck, skip connections
    """
    def __init__(self, in_channels=4, base_channels=32):
        super(SiameseUNet, self).__init__()

        # Shared encoder (Siamese)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (concatenates T1 and T2 features)
        self.bottleneck = DoubleConv(base_channels * 8 * 2, base_channels * 16)

        # Decoder (skip connections from both T1 and T2)
        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 8 * 3, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4 * 3, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 2 * 3, base_channels * 2)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 3, base_channels)

        # Output layer
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, t1, t2):
        # T1 encoder path
        e1_t1 = self.enc1(t1)
        e2_t1 = self.enc2(self.pool1(e1_t1))
        e3_t1 = self.enc3(self.pool2(e2_t1))
        e4_t1 = self.enc4(self.pool3(e3_t1))
        bottleneck_t1 = self.pool4(e4_t1)

        # T2 encoder path (shared weights)
        e1_t2 = self.enc1(t2)
        e2_t2 = self.enc2(self.pool1(e1_t2))
        e3_t2 = self.enc3(self.pool2(e2_t2))
        e4_t2 = self.enc4(self.pool3(e3_t2))
        bottleneck_t2 = self.pool4(e4_t2)

        # Concatenate bottleneck features
        bottleneck = torch.cat([bottleneck_t1, bottleneck_t2], dim=1)
        bottleneck = self.bottleneck(bottleneck)

        # Decoder with skip connections from both T1 and T2
        d1 = self.up1(bottleneck)
        d1 = torch.cat([d1, e4_t1, e4_t2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3_t1, e3_t2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2_t1, e2_t2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1_t1, e1_t2], dim=1)
        d4 = self.dec4(d4)

        logits = self.out(d4)

        return logits
