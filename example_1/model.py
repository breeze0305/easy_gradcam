import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)


        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(32 * 56 * 56, 128)
        self.linear2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.flatten(1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
