import torch
from torch import nn
from torch.utils.data import Dataset

class MyConvDataset1Layer(Dataset):
    def __init__(self, X_, y_):
        super(Dataset, self).__init__()
        self.X = torch.tensor(X_.values,dtype=torch.float32)
        self.y = torch.tensor(y_.values,dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].reshape((5, 5)).unsqueeze(0), self.y[index]

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, padding='same')
        self.bn = nn.BatchNorm2d(32)

        self.head = nn.Linear(32 * 8 * 8, 128)
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        out = self.head(out)
        return out.reshape((128, -1))