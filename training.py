import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt
import time

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)

    running_loss = 0.0

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device)  # all data & model on same device

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        #print("Batch ",batch_idx,"finish")
        if batch_idx%20000 == 0:
            print("batch idx",batch_idx)

    end_time = time.time()

    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
    return running_loss


def validate_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
    
def predict_model(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        prediction = []

        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            prediction += predicted.cpu().numpy().flatten().tolist()

        return prediction