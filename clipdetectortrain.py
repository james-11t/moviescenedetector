from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils, datasets
from torchvision.models import resnet18, ResNet18_Weights
import os
import torch.optim as optim


data_transforms = {
    'train': transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224, 0.225])
        ]
    ),
     'val': transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224, 0.225])
        ]
    ),
}

#Preprocessing the data so that it can be handled by the model

data_dir = 'dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}


#We are using ImageFolder in order to load the images from the data set

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size =4, shuffle = True, num_workers = 2) for x in ['train','val']}


model = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features,2)

#Here we are loading the resnet18 model so it can be utilised for binary classification

for name, param in model.named_parameters():
    if "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

#Loops through all the parameters in the neural network. And only trains parameters that belong to the fully connected layer.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#If GPU is available, this will be used for training, otherwise, the CPU will be utilised

total = 8560
veo3weight = total / 2 / 1520
realweight = total / 2/ 7040
class_weights = torch.tensor([veo3weight, realweight]).to(device)
#There is an imbalance between both data sets
#As a result, we perform a calculation so the model focuses more on the minority class
#This will prevent bias, and will also prevent the model from cheating

criterion = nn.CrossEntropyLoss(weight = class_weights)
#Loss function used to calculate error during training

class_weights = torch.tensor([veo3weight, realweight], dtype=torch.float32).to(device)



optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) 
#This is the algorithm that updates the model weights during training

model = model.to(device)


#Below, we begin to train the model

if __name__ == '__main__':
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    num_epochs = 10

    #This means the model will run through the whole data set 10 times

    for epoch in range(num_epochs):
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs,labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #This calculates the loss and accuracy for each batch of data
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            

            print (f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc: .4f}')

            #The print statement will let us know the accuracy after each epoch, as well as the epoch loss

            #We want to know the accuracy on the training data, and the accuracy on the validation data

torch.save(model.state_dict(), "clipdetectormodel.pth")

#Save the model so we can utilise it to classify unseen images