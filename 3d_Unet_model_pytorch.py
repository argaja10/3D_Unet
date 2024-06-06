import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        self.encoder5 = self.conv_block(128, 256)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.upconv4 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(128, 64)
        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(32, 16)
        
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return out

# Test the model
model = UNet3D(3, 4)
x = torch.randn((1, 3, 128, 128, 128))
output = model(x)
print(output.shape)
