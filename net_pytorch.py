import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.features = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x, angle):
        x = self.features(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sig(x)
        # print(x.size())
        return x

net = MyNet()
net(Variable(torch.zeros(2,3,75,75).float()), 0)
