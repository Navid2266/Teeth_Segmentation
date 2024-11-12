import torch
import torch.nn as nn
from torchvision import models

class SegNetVGG16(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(SegNetVGG16, self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # Freeze VGG16 layers
        for layer in features:
            for param in layer.parameters():
                param.requires_grad = False

        # Encoder blocks (using VGG16 layers)
        self.encoder_block1 = nn.Sequential(*features[:4])    # block1 (conv1 + conv2)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.encoder_block2 = nn.Sequential(*features[5:9])   # block2
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.encoder_block3 = nn.Sequential(*features[10:16]) # block3
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.encoder_block4 = nn.Sequential(*features[17:23]) # block4
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.encoder_block5 = nn.Sequential(*features[24:30]) # block5
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder blocks (mirroring the encoder)
        self.upconv5 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.decoder_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1)
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.output_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        x1 = self.encoder_block1(x)
        x1, idx1 = self.pool1(x1)
        
        x2 = self.encoder_block2(x1)
        x2, idx2 = self.pool2(x2)
        
        x3 = self.encoder_block3(x2)
        x3, idx3 = self.pool3(x3)
        
        x4 = self.encoder_block4(x3)
        x4, idx4 = self.pool4(x4)
        
        x5 = self.encoder_block5(x4)
        x5, idx5 = self.pool5(x5)

        # Decoding path
        x5 = self.upconv5(x5)
        x5 = self.decoder_block5(x5)
        
        x4 = self.upconv4(x5)
        x4 = self.decoder_block4(x4)
        
        x3 = self.upconv3(x4)
        x3 = self.decoder_block3(x3)
        
        x2 = self.upconv2(x3)
        x2 = self.decoder_block2(x2)
        
        x1 = self.upconv1(x2)
        x1 = self.decoder_block1(x1)

        # Final output layer
        output = self.output_conv(x1)
        
        return output

# Instantiate the model and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegNetVGG16(input_channels=3, output_channels=1).to(device)

# Check model summary (optional)
print(model)
