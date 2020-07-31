"""
    The predict.py script successfully reads in an image and a checkpoint 
    then prints the most likely image class and it's associated probability.
    
    args:
       
        path user enters a directory path like /path/to/image. 
        Default for testing is flowers/test/83/image_01742.jpg.
        
        checkpoint user enters name of checkpoint. Default is july_checkpoint.pth.
    
        --top_k allows users to print out the top K classes along with 
        associated probabilities. Default is 5.
        
        --category_names allows users to load a JSON file that maps the class values 
        to other category names. Default is cat_to_name.json.
        
        --gpu allows users to use the GPU to calculate the predictions, if
        available. Default is true.
        
    Command Line:
    
        python predict.py /path/to/image checkpoint --top_k --category_names --gpu
        example: python predict.py /path/to/image checkpoint --top_k 7 --gpu False
    
    Output:
    
        Topk predicted image classes and probabilities.
        
        The actual image class.
        
        The most likely image class with its probability.
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

parser.add_argument('path', type=str, default='flowers',
                    help='Enter location of the directory with images.')
parser.add_argument('checkpoint', type=str, default='july_checkpoint.pth',
                    help='Enter name of file with checkpoint.')
parser.add_argument('--top_k', action='store', type=int, default=5,
                    help='Number of top predictions, default is 5.')
parser.add_argument('--category_names', action='store', type=str,
                    default='cat_to_name.json', 
                    help='Enter a JSON file to replace cat_to_name.json.')
parser.add_argument('--gpu', action='store', default=True,
                    help='Use GPU to predict? True = GPU, False = CPU.')

args = parser.parse_args()

# assign checkpoint file
checkpoint = args.checkpoint

# assign topk value
topk = args.top_k

# assign a JSON file that maps the class values to categories
category_names = args.category_names
short_json = category_names[:-5]

# load label to flower image dictionary
with open(category_names, 'r') as f:
    short_json = json.load(f)
    
img = args.path
print(img)

# # set directories for flower folders
# data_dir = 'flowers'
# train_dir = data_dir + '/train'
# valid_dir = data_dir + '/valid'
# test_dir = data_dir + '/test'

# # generate a path to a random image, call process_image()
# # check name of image, label
# global img
# img = random.choice(glob.glob(test_dir +'/*/*.jpg')) 
# print(img)

# find the folder number of flower folder
if img[-19] == '/':
    # 2 digit folder number
    category = img[-18:-16]
elif img[-18] == '/':
    # 1 digit folder number
    category = img[-17:-16]
else:
    # 3 digit folder number
    category = img[-19:-16]
    
# function that loads a checkpoint and rebuilds the model
def load_trained_model(filepath):
    ''' Load a trained model.
    '''
    state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if state['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        model.class_to_idx = state['class_to_idx']
        model.classifier = state['classifier']  
        model.load_state_dict(state['model_state_dict'])
        return model

model = load_trained_model(checkpoint)
    
def process_image(image_path):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.  
    '''
    img = Image.open(image_path)
    
    # get dimensions of the image
    width, height = img.size

    # resize image, keeping aspect ratio, 256 shortest size
    if width < height:        
        new_height = int(256 * (height/width))
        resized_image = img.resize((256, new_height))        
    else:
        new_width = int(256 * (width/height))
        resized_image = img.resize((new_width, 256))
    
    width, height = resized_image.size
            
    # centercrop to 224 X 224     
    crop_width = 224
    crop_height = 224
    left = (width - crop_width)/2
    top = (height - crop_height)/2
    right = (width - crop_width)/2 + crop_width 
    bottom = (height - crop_height)/2 + crop_height    
    resized_image = resized_image.crop((left, top, right, bottom))
    width, height = resized_image.size

    # turn image into array
    array_image = np.array(resized_image) 
    
    # turns all RGB color values into range(0:1)
    array_image = array_image/255
    
    # normalize and transpose images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    array_image = (array_image - mean) / std     
    array_image = array_image.transpose((2, 0, 1))

    # modify the output of process_image function to a tensor
    processed_image = torch.from_numpy(array_image)
    tensor_image = processed_image.float()
    
    return tensor_image 

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # send model to device, either cuda or cpu
    # Use GPU if available unless user doesn't want
    if args.gpu == True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print('Program is using ' + str(device) + '.\n') 

    # move model into evaluation mode
    model.eval()

    # process image
    pred_image = process_image(image_path) 
    pred_image = pred_image.to(device)
    
    # add batch size
    pred_image = pred_image.unsqueeze(0)
    
    # run model
    with torch.no_grad():
        output = model.forward(pred_image)
    
    # calculate probabilities
    #ps = torch.exp(output)
    probability = F.softmax(output.data,dim=1) 
    
    # find top k probabilities and the indexes
    probs, classes = torch.topk(probability, dim=1, k=topk)
    
    # change probs & classes from tensor to list
    probs = probs.tolist()[0]
    classes = classes.tolist()[0] 
    
    # initialize an array for flower names
    global flower_names
    flower_names = []
    
    print('The top ' + str(topk) + ' image classes and probabilities: \n')
    # find corresponding flower names in cat_to_name
    i = 0
    while i < topk: 
        label = classes[i]
        lookup = (model.class_to_idx.get(label))
        flower_names.append(short_json[lookup])
        print(flower_names[i], probs[i])
        i += 1
    return probs, classes

probs, classes = predict(img, model, topk)

def main():
    print('')
    print('The actual image class is ' + str(short_json[category]) + '.\n')
    print('The most likely image class is ' + str(flower_names[0]))
    print('   with a probability of ' + str(probs[0]) + '.')
    print('')


if __name__ == '__main__':
    main()