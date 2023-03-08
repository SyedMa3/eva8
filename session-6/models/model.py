import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
          ) # 32>32 | 3

        self.transblock1 = nn.Sequential(
          nn.Conv2d(32, 32, 3, stride=2), # 15>7| 13
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout(dropout_value),
        )


        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
          ) # 15>15 | 9
        self.transblock2 = nn.Sequential(
          nn.Conv2d(32, 32, 3, stride=2), # 15>7| 13
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout(dropout_value),
        )


        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value)
          ) # 7>7 | 21
        self.transblock3 = nn.Sequential(
          nn.Conv2d(32, 32, 3, stride=2), # 15>7| 13
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Dropout(dropout_value),
        )


        self.convblock4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),

            nn.Conv2d(64, 10, 3, padding=1, bias=False),
            nn.ReLU(),
          ) # 3>3 | 45

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # 1 | 61

        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.transblock1(self.convblock1(x))
        x = self.transblock2(self.convblock2(x))
        x = self.transblock3(self.convblock3(x))
        x = (self.convblock4(x))
        x = self.gap(x)
        x = x.view(-1, 10)

        x = self.fc(x)
        return F.log_softmax(x, dim = -1)