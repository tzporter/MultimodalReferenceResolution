import torch.nn as nn

# Import your base model definition.
# Ensure 'base_model.py' is in a location Python can find, like the project root.
from base_model import MultimodalPoseAudioTransformer

class ClassificationHead(nn.Module):
    """Simple MLP to run on top of the PCA-reduced embeddings."""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)