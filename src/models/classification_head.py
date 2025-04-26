import torch.nn as nn
import segmentation_models_pytorch as smp

class ClassificationHead(nn.Module):
    def __init__(self, encoder_name, num_classes):
        super().__init__()
        self.encoder_name = encoder_name
        if encoder_name.startswith("tu-"):
            self.encoder = smp.encoders.get_encoder(encoder_name, weights="imagenet")
        else:
            self.encoder = smp.encoders.get_encoder(encoder_name, pretrained=True)
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Global Pooling
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_classes)

    def forward(self, x):
        features = self.encoder(x)  # Get feature maps
        x = features[-1]  # Take last layer's output
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # Pool and flatten
        x = self.fc(x)  # Fully connected layer
        return x
