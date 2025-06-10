# 2025 Caleb Mostyn

import torch.nn as nn

class SimpleNet(nn.Module):
    """
    A simple linear classifier for the AT&T face dataset (112x92 grayscale).
    The simplicity of the model is intentional to facilitate easier inversion.
    """
    def __init__(self, input_features=112 * 92, output_features=40):
        super(Net, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.regression = nn.Linear(
            in_features=self.input_features, out_features=self.output_features
        )

    def forward(self, x):
        x = x.view(-1, self.input_features)
        x = self.regression(x)
        return x
