import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_image):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_image, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
