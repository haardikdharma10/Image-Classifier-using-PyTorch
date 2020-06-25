#Importing necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# Defining Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = 'Option to use GPU', type = str)

#Setting values data loading
args = parser.parse_args ()

#data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#Loading Data
if data_dir:
    train_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    validation_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder (train_dir, transform = train_transforms)
    validation_data = datasets.ImageFolder (valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder (test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(validation_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)

#Mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):
    if arch == 'vgg16': 
        model = models.vgg16 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: 
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #If hidden_units are not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    
    model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch

# Defining validation Function. will be used during training
def validation(model, validationloader, criterion):
    model.to (device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in validationloader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

#Loading model using above defined functiion
model, arch = load_model (args.arch, args.hidden_units)

#Initializing criterion and optimizer
criterion = nn.NLLLoss ()
if args.lrn: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to (device) #device can be either cuda or cpu
#setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:
    epochs = 8

print_every = 10
steps = 0

#runing through epochs
for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (trainloader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () 

        # Forward and backward passes
        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) #calculating loss (cost function)
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item () # loss.item () returns scalar value of Loss function

        if steps % print_every == 0:
            model.eval ()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validationloader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validationloader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(validationloader)*100))

            running_loss = 0
            # Make sure training is back on
            model.train()

#saving trained Model
model.to ('cpu')
# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
#saving trained model for future use
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')
