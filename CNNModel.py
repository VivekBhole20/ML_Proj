# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:58:57 2024

@author: vicky
"""

import torch
from torch import nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1=nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv2=nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv3=nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
            )
        self.conv4=nn.Sequential(
            nn.Dropout2d(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.flatten=nn.Flatten()
        self.linear1=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*8*26,out_features=10000,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(10000)
            )
        self.linear2=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=10000,out_features=3000,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(3000)
            )
        self.linear3=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=3000,out_features=1000,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(1000)
            )
        self.linear4=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1000,out_features=256,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256)
            )
        self.linear5=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256,out_features=10,bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(10)
            )
        self.output=nn.Linear(in_features=10,out_features=1)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,input_data):
        x=self.conv1(input_data)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        x=self.linear4(x)
        x=self.linear5(x)
        x=self.output(x)
        x=self.sigmoid(x)
        
        return x

        
if __name__=="__main__":
    x=torch.ones((20,1,90,431))
    model=CNNModel()
    print(model(x).shape)

    """from torch.utils.data import Dataset,DataLoader,Subset,ConcatDataset
    import os
    import numpy as np
    from scipy.stats import loguniform

    from sklearn.model_selection import RandomizedSearchCV
    from skorch import NeuralNetBinaryClassifier
    from skorch.helper import SliceDataset
    error=os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print(error)
    from AudioLoader import train_dataset,valid_dataset

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(1)

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    params={
        'optimizer__lr' : loguniform(0.0000001,1),
        'batch_size' : [20,35,50],
        'max_epochs' : [10,20,30]
    }

    #dataset=ConcatDataset([train_dataset,valid_dataset])
    dataset=Subset(train_dataset,torch.arange(5000))

    print(device)
    model=NeuralNetBinaryClassifier(
        CNNModel,
        criterion=nn.BCELoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        #dataset=dataset,
        train_split=False,
        verbose=0,
        device=device
    )

    

    X=SliceDataset(dataset,idx=0)
    y=SliceDataset(dataset,idx=1)

    #y=y.view(-1,1)
    #y=np.reshape(-1,1)#
    print(X)
    print(y)
    #model.fit(X,y)
    #print("model fitted")

    #grid_search=GridSearchCV(model,params,scoring="accuracy",n_jobs=-1,refit=True,cv=7,verbose=1,error_score="raise")
    #results=grid_search.fit(X,y)

    rs=RandomizedSearchCV(estimator=model,param_distributions=params,scoring="accuracy",refit=True,n_iter=5,cv=5,random_state=1,n_jobs=1)

    results=rs.fit(X,y)

    print("Best: %f using %s" % (results.best_score_, results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))"""