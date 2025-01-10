import torch
from torch import nn

class CNN1DModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Dropout1d(p=0.1),
            nn.Conv1d(in_channels=90,out_channels=90,kernel_size=5,padding=2),
            nn.BatchNorm1d(90),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )
        self.conv2=nn.Sequential(
            nn.Dropout1d(p=0.25),
            nn.Conv1d(in_channels=90,out_channels=128,kernel_size=5,padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        )

        self.flatten=nn.Flatten()
        self.linear1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*47,out_features=1200),
            nn.ReLU()
        )
        self.linear2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1200,out_features=20),
            nn.ReLU()
        )
        self.output=nn.Linear(in_features=20,out_features=1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.output(x)
        x=self.sigmoid(x)

        return x

if __name__=="__main__":
    x=torch.ones((20,90,431))
    model=CNN1DModel()
    print(model(x).shape)