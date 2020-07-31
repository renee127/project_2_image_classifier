"""
    The train.py successfully trains a new network on a dataset of images 
    and saves the model to a checkpoint.
    
    Print the training loss, validation loss, and validation accuracy.
    
    args:
    
        data_dir = directory with the images for the classifier to train. 
        
        --save_dir allows user to choose where to save checkpoint. Default
        is current directory.
       
        --arch allows users to choose from at least two different 
        architectures available from torchvision. Default is vgg16.
        
        --learning_rate allows users to set hyperparameter for learning 
        rate. Default is 0.01.
        
        --hidden_units allows users to set hyperparameter for number of 
        hidden units. Default is 512.
        
        --epochs allows users to set hyperparameter for training epochs. 
        Default is 20.
        
        --gpu allows users to use the GPU to calculate the predictions, if
        available. Default is True.

    Command Line:
        
        data_dir = directory with the images for the classifier to train. Use
        flowers by default.
        
        python train.py data_dir --arch --learning_rate --hidden_units --epochs --gpu
        example: python train.py data_dir --arch vgg16 --hidden_units 8000 
    
    Output:
    
        Prints out training loss, validation loss, and validation accuracy.
        
        Saves a checkpoint.
"""

# Imports

import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import json
import torch
import torch.nn as nn
import torch.optim as optim
from workspace_utils import active_session
from PIL import Image 
import helper
import glob
import random
from matplotlib import pyplot as plt
import torch.nn.functional as F
import argparse

# use argparse to get command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default='flowers',
                    help='Enter location of the directory with images.')
parser.add_argument('--save_dir', action='store', type=str,
                    default='', 
                    help='Enter a directory to save checkpoint.')
parser.add_argument('--arch', action='store', type=str, default='vgg16',
                    help='Choose vgg16 or densenet161.')
parser.add_argument('--learning_rate', action='store', type=float, default=0.01,
                    help='Enter a float number for learning rate.')
parser.add_argument('--hidden_units', action='store', type=int, default=512,
                    help='Enter an integer for hidden units.')
parser.add_argument('--epochs', action='store', type=int, default=20,
                    help='Enter an integer for epochs.')
parser.add_argument('--gpu', action='store', default=True,
                    help='Use GPU to predict? True = GPU, False = CPU.')

args = parser.parse_args()

print(args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

path = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
# load label to flower image dictionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# set directories for image folders
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# transforms applied to 3 datasets

# training data resized, randomly cropped, 
# randomly flipped, normalized mean and std
# ToTensor converts image into numbers
train_transforms = transforms.Compose([
    transforms.RandomRotation(27),
    transforms.Resize(255),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# validation data resized with normalized mean and std
# ToTensor converts image into numbers
valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# test data resized with normalized mean and std
# ToTensor converts image into numbers
test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the datasets with ImageFolder and apply transforms
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# load images using dataloder from torch library
# batching allows dataloader automatically fetch data samples
# shuffle=True the data gets “shuffled” before each epoc

train_loaders = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loaders = DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loaders = DataLoader(test_dataset, batch_size=16, shuffle=True)

# verifying items in list
print(str(len(cat_to_name)) + ' categories of flowers')
print(json.dumps(cat_to_name, indent=1, sort_keys=False))

# Use torchvision to import pretrained model
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    in_units = 25088
elif arch == 'densenet161':
    model = models.densenet161(pretrained=True)
    in_units = 2208

# print model summary to get default input number
print(model.classifier)

# protect the convolutional layer from getting trained
# freeze parameters so don't backprop though them
for param in model.parameters():
    param.requires_grad = False
    
# create new feed-forward network as a classifier
# using ReLU activations, dropout, hidden layers
new_classifier = nn.Sequential(nn.Linear(in_units, hidden_units), 
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim=1)
                           )

# replace dafault classifier with new classifier
model.classifier = new_classifier

# move model to GPU device, if available,
# confirm device usage
if args.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); 
    print(f'{device} being used for training.')
    
# calculate error with negative log likelihood loss
criterion = nn.NLLLoss()

# train the classifier parameters, feature parameters are frozen
# set optimizer (SGA or ADAM) with parameters (will update every round)
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate )

# train the model and evaluate the model 

# keeps session active during long-running work
with active_session():
    # move model to GPU device, if available,
    # confirm device usage
    if args.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); 
        print(f'{device} being used for training.')
        
    # train the classifier layers using backpropagation 
    # using the pre-trained network to get the features
    
    # set value to switch from model.train() to eval.train()
    print_every = 150
    
    # initialize the lists for graphing
    train_losses = []
    valid_losses = []

    print('Training started')
    for epoch in range(epochs):
        sum_of_train_losses = 0 
        accuracy = 0
        
        # training the model
        model.train()
        steps = 0
        
        for images, labels in train_loaders:
            steps += 1
            
            # move input and label tensors to CPU or GPU
            images, labels = images.to(device), labels.to(device)
            
            # clears weights/gradients before data parsing
            optimizer.zero_grad()
            
            # forward pass
            logps = model.forward(images)
            
            # calculate loss
            loss = criterion(logps, labels)
            
            # backpropogation
            loss.backward()
            
            # adjust parameters based on gradients
            optimizer.step()
            
            # add the loss to the running_total
            sum_of_train_losses += loss.item()
            
            print(f"Training loss: {loss:.3f}.. ")
            
            # run model.eval every 50 cycles (print_every) and print loss/accuracy
            if steps % print_every == 0:
            
                # evaluate the model
                model.eval()
                sum_of_valid_losses = 0 # running total of losses in evaluation
                steps = 0
                accuracy = 0

                # don't calculate gradients
                with torch.no_grad():

                    for images, labels in valid_loaders:
                        steps += 1

                        # move input and label tensors to CPU or GPU
                        images, labels = images.to(device), labels.to(device)

                        # forward pass
                        logps = model.forward(images)

                        # calculate loss, running total (test_loss)
                        batch_loss = criterion(logps, labels)
                        sum_of_valid_losses += batch_loss.item()

                        # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                # collect values in lists for graph to check for overfitting                    
                train_losses.append(sum_of_train_losses/print_every)
                valid_losses.append(sum_of_valid_losses/len(valid_loaders))                               
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {sum_of_train_losses/print_every:.3f}.. "                                
                      f"Valid loss: {sum_of_valid_losses/len(valid_loaders):.3f}.. "                
                      f"Valid accuracy: {accuracy/len(valid_loaders)*100.:.2f} %")
                # reset running total of training loss since 
                sum_of_train_losses = 0
                sum_of_valid_losses = 0
                model.train()
# evaluate the model against test data

test_loss = 0
steps = 0
accuracy = 0

# don't calculate gradients
with torch.no_grad():
    print('Testing model')
    model.eval()
    
    # move model to GPU device, if available,
    # confirm device usage
    if args.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device); 
        print(f'{device} being used for training.')     

    for images, labels in test_loaders:
        steps += 1

        # move input and label tensors to CPU or GPU
        images, labels = images.to(device), labels.to(device)

        # forward pass
        logps = model.forward(images)

        # calculate loss
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        # calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(test_loaders):.3f}.. "
    f"Test accuracy: {accuracy/len(test_loaders)*100.:.2f} %")
    
print('Evaluation finished')

# create and save a checkpoint
class_to_idx = train_dataset.class_to_idx
model.class_to_idx = { class_to_idx[k]: k for k in class_to_idx}

state = {
    'batch_size' : 16,
    'lr' : learning_rate,
    'epoch' : epochs,
    'arch' : arch,
    'input_size' : in_units,
    'output_size' : 102,
    'model_state_dict' : model.state_dict(),
    'classifier' : model.classifier,
    'class_to_idx' : model.class_to_idx
}
# save checkpoint in directory with file name checkpoint.pth

new_path=str(path) + '\checkpoint.pth'
torch.save(state, new_path)

def main():
    print('Training is finished.')

if __name__ == '__main__':
    main()