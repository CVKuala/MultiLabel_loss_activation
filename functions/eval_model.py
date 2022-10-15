import numpy as np
import pandas as pd
import torch

def evaluate(model, data_loader, device):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_true = np.array([[]], dtype=np.int)
    y_pred = np.array([[]], dtype=np.int)
    
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(y_true)
            print(labels.cpu())
            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, predicted.cpu()))
    
    error = np.sum(y_pred != y_true) / len(y_true)
    return error
