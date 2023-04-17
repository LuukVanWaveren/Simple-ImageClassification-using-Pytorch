
## Data science tools
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,nrClasses):
        super(Net,self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(12800, 1280)
        self.fc2 = nn.Linear(1280, 160)
        self.fc3 = nn.Linear(160, nrClasses)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool1(F.relu(self.conv4(x)))
        x = x.view(-1, 12800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x