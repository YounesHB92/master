import segmentation_models_pytorch as smp
import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self, encoder_name, num_classes, dropout_prob=0.5):
        """
        A classification head using a pretrained encoder with global average pooling and fully connected layer.

        Args:
            encoder_name (str): Encoder model name from segmentation_models_pytorch.
            num_classes (int): Number of output classes.
            dropout_prob (float): Dropout probability after pooling. Default is 0.5.
        """
        super().__init__()
        self.encoder_name = encoder_name

        if encoder_name.startswith("tu-"):
            self.encoder = smp.encoders.get_encoder(encoder_name, weights="imagenet")
        else:
            self.encoder = smp.encoders.get_encoder(encoder_name, pretrained=True)

        self.pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.encoder.out_channels[-1], num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

        print("\nCnn model initialized successfully.")

    def forward(self, x):
        features = self.encoder(x)  # list of intermediate features
        x = features[-1]  # use the last feature map
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # [B, C, 1, 1] → [B, C]
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x
