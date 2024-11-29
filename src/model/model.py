import torch.nn as nn
import torchvision.models as models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class OCTModel(nn.Module):
    def __init__(self, num_classes=28):
        super(OCTModel, self).__init__()
        logging.info("Initializing the OCT Model...")
        # Load ResNet50
        self.base_model = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        num_features = self.base_model.fc.in_features

        # Replace the final fully connected layer with a custom classifier
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        logging.info("Model initialized successfully!")

    def forward(self, x):
        return self.base_model(x)
