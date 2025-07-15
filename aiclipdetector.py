import torch
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


model = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('clipdetectormodel.pth'))
model.eval()

#The model has been trained, now we load it so we can utilise it for unseen data

image_path = input('Enter the path to your image file ')
try:
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
    ])

    #This allows us to preprocess the data


    input_tensor = preprocess(image)
    inputbatch = input_tensor.unsqueeze(0) #Here we are adding a batch dimension

    with torch.no_grad():
        output = model(inputbatch)
     #Speeds up model processing as it has already been trained

    prob = F.softmax(output, dim = 1)
    _, predictedclass = output.max(1)
    classnames = ['real','AI generated']
    #These are the two classes images can be classified into
    predicted_class_name = classnames[predictedclass.item()]
    values = prob.flatten().tolist()
    state = ''
    maxvalue = max(values)
    if maxvalue < 0.9:
        state = 'Error - Could not identify image'
    #If the model is not certain that an image belongs to a particular class, it is likely it doesn't belong to any. As a result we throw an error to the user
    #If the probability of it belonging to a class is above 0.9, we know the model is certain it belongs to a given class.
    else:
        state = f'This movie scene snapshot is likely to be {predicted_class_name}'
    print (state)
except FileNotFoundError:
    print ('File not found, please check path')
except Exception as e:
    print (f'Error loading image {e}')

#The user will be prompted to input an image, if this cannot be found on their system, an error will be raised
