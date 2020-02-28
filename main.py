import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt
import time

from wsj_loader import *
from models import *
from training import *
import pandas as pd

k_context = 12

cuda = torch.cuda.is_available()

if cuda:
    MyDevice = torch.device('cuda')
    os.environ['WSJ_PATH'] = os.environ['HOME'] + r'/hw1/float32/'
else:
    MyDevice = torch.device('cpu')
    os.environ['WSJ_PATH'] = os.environ['HOME'] + r'/Desktop/11785 Introdcution to Deep Learning/hw1p2/float32/'
    print(os.environ['WSJ_PATH'])

num_workers = 0 if sys.platform == 'win32' else 0

loader = WSJ()

# Training
train_dataset = MyDataset(loader.train, k=k_context, mydevice=MyDevice)

train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=True, batch_size=10)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# validation
validation_dataset = MyDataset(loader.dev, k=k_context, mydevice=MyDevice)

validation_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
validation_loader = data.DataLoader(validation_dataset, **validation_loader_args)

# testing
test_dataset = MyDataset(loader.test, k=k_context, mydevice=MyDevice)

test_loader_args = dict(shuffle=False, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda\
                    else dict(shuffle=False, batch_size=1)
test_loader = data.DataLoader(test_dataset, **test_loader_args)


model = Simple_MLP([(k_context*2+1)*40, 1024, 1024, 1024, 512, 512, 512, 512, 138])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
print(model)

n_epochs = 20
Train_loss = []
Test_loss = []
Test_acc = []
prediction = []

for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, MyDevice)
    test_loss, test_acc = validate_model(model, validation_loader, criterion, MyDevice)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    prediction.append(predict_model(model, test_loader, MyDevice))
    df = pd.DataFrame({"label":prediction[-1]})
    df.to_csv(r"deepresult32_" + str(i) + r".csv")
    print('='*20)
    






