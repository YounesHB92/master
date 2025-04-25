import torch.nn as nn
import segmentation_models_pytorch as smp

# Load a pretrained encoder
encoder_name = "efficientnet-b0"
encoder = smp.encoders.get_encoder(encoder_name, pretrained=True)

# Define a classification model
class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Global Pooling
        self.fc = nn.Linear(encoder.out_channels[-1], num_classes)

    def forward(self, x):
        features = self.encoder(x)  # Get feature maps
        x = features[-1]  # Take last layer's output
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # Pool and flatten
        x = self.fc(x)  # Fully connected layer
        return x

# Initialize model
model = ClassificationModel(encoder, num_classes)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Print model summary
print(model)