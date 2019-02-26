import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.autograd import Variable
from shutil import copyfile

im_size = 224
lab = ['aircrafts', 'birds', 'cars', 'dogs', 'flowers']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('models/coarse_grained_vgg16.pth')


test_transforms = transforms.Compose([transforms.Resize([im_size,im_size]),
                                      transforms.ToTensor()])

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inputa = Variable(image_tensor)
    inputa = inputa.to(device)
    output = model(inputa)
    index = output.data.cpu().numpy().argmax()
#     print(index)
    return index



for img in os.listdir("final_test_data"):
#     for i in os.listdir("coarse_test/" + category):

    image = Image.open("final_test_data/" + img)
    index = predict_image(image)
    coarse_class = lab[index]
    src = "final_test_data/" + img
    directory = "bilinear-cnn/data/testing/" + coarse_class
    if not os.path.exists(directory):
        os.makedirs(directory)
    dest = "bilinear-cnn/data/testing/" + coarse_class + "/" + img
    copyfile(src, dest)
    
# print("Test accuracy : ", count/total)
# print(total)