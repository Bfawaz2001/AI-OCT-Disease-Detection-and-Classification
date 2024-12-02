import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class OCTModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.0):
        super(OCTModel, self).__init__()
        # Load pre-trained EfficientNet-B0
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify the classifier to include a dropout layer and adjust the output size
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Add dropout
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
