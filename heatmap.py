from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse
import time

from PIL import Image
#import torchvision.models as models
from utils import progress_bar
from torch.autograd import Variable
import requests
import seaborn as sns;
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math



class Two_Branch_Net(nn.Module):
	def __init__(self):
		super(Two_Branch_Net, self).__init__()

		self.base_model = vgg11_head

		self.classifier = nn.Sequential(
			nn.Conv2d(512, 4096, kernel_size=7),
			nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
			nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 16, kernel_size=1)
			)

		self.classifier_shape = nn.Sequential(
			nn.Conv2d(512, 4096, kernel_size=7),
			nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
			nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 6, kernel_size=1)
			)

	def forward(self, x):
		out = self.base_model(x)
		# out = out.view(out.size(0), -1)
		out1 = self.classifier(out)
		out2 = self.classifier_shape(out)
		print(out1)
		out1 = out1.view(out1.size(1), out1.size(2)*out1.size(3))
		out2 = out2.view(out2.size(1), out2.size(2)*out2.size(3))

		return out1, out2


imsize = 700
imwidth = 640
imheight = 480

sns.set()
use_cuda = torch.cuda.is_available()
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

print("=====>Loading model...")
# vgg11 = models.vgg11(pretrained=True)

convnet = [[3,1,1], [2,2,0], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [7,1,0], [1,1,0], [1,1,0]]
layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'pool3','conv5', 'conv6', 'pool4', 'conv7', 'conv8', 'pool5', 'conv9', 'conv10', 'conv11']


#class definition for fully convolutional vgg11
#vgg11_head contains all the convolutional layers in vgg11
#then fully_conv_vgg append three more convolutional layers to vgg11_head
# vgg11_head = nn.Sequential(*list(vgg11.children())[:-1])
# class Fully_Conv_Vgg(nn.Module):
# 	def __init__(self):
# 		super(Fully_Conv_Vgg, self).__init__()

# 		self.base_model = vgg11_head
# 		self.conv1 = nn.Conv2d(512,4096, kernel_size=7)
# 		self.conv2 = nn.Conv2d(4096,4096, kernel_size=1)
# 		self.conv3 = nn.Conv2d(4096, 1000, kernel_size=1)

# 	def forward(self, x):
# 		out = self.base_model(x)
# 		print(out)
# 		out = self.conv1(out)
# 		out = self.conv2(out)
# 		out = self.conv3(out)

# 		return out

# model = Fully_Conv_Vgg()
# #load the pretrained coefficient
# model.base_model.load_state_dict(vgg11_head.state_dict())
# #reshape the pre-trained fc layer coefficient into convolutional layer coeffcient form
# model.conv1.load_state_dict({"weight": vgg11.classifier[0].state_dict()["weight"].view(512, 4096,7,7),
# 	                         "bias":vgg11.classifier[0].state_dict()["bias"]})
# model.conv2.load_state_dict({"weight": vgg11.classifier[3].state_dict()["weight"].view(4096, 4096,1,1),
# 	                         "bias":vgg11.classifier[3].state_dict()["bias"]})
# model.conv3.load_state_dict({"weight": vgg11.classifier[6].state_dict()["weight"].view(1000, 4096,1,1),
# 	                         "bias":vgg11.classifier[6].state_dict()["bias"]})

checkpoint = torch.load('./checkpoint/model_vgg11.ta')
model = checkpoint['net']



if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

#preprocess crop the image at the center by imsize; this can be changed but notice that it will result in different output size
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


#preprocess2 is the same as preprocess but remove the normalization for visualization
preprocess2 = transforms.Compose([
    transforms.ToTensor()
])

#receptive field size 
# j(out) = j(in)*s
# r(out) = r(in) + (k-1)*j(in)
# start(out) = start(in) + ((k-1)/2-p)*j(in)
# maxpool layer is filter with kernal size 2 and stride 2
# start(in) + 1*1 + 1*2 + 1*4 + 1*4 + 1*8 + 1*8 + 1*16+1*16 = 59


def outFromIn(conv, layerIn):
  n_in = layerIn[0]
  j_in = layerIn[1]
  r_in = layerIn[2]
  start_in = layerIn[3]
  k = conv[0]
  s = conv[1]
  p = conv[2]
  
  n_out = math.floor((n_in - k + 2*p)/s) + 1
  actualP = (n_out-1)*s - n_in + k 
  pR = math.ceil(actualP/2)
  pL = math.floor(actualP/2)
  
  j_out = j_in * s
  r_out = r_in + (k - 1)*j_in
  start_out = start_in + ((k-1)/2 - pL)*j_in
  return n_out, j_out, r_out, start_out
  
def printLayer(layer, layer_name, ):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
 
def receptive_field(idx_x, idx_y):
  layerInfos=[]
  #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
  print ("-------Net summary------")
  currentLayer = [imwidth, 1, 1, 0.5]
  currentLayer2 = [imheight, 1, 1, 0.5]
  printLayer(currentLayer, "input image")
  for i in range(len(convnet)):
    currentLayer = outFromIn(convnet[i], currentLayer)
    currentLayer2 = outFromIn(convnet[i], currentLayer2)
    layerInfos.append(currentLayer)
    printLayer(currentLayer, layer_names[i])
  print ("------------------------")

  n = layerInfos[-1][0]
  j = layerInfos[-1][1]
  r = layerInfos[-1][2]
  start = layerInfos[-1][3]
  start_2 = currentLayer2[3]
  j_2 = currentLayer2[1]

  #return the center coordinate of the receptive field with its corresponding size
  return (start+idx_x*j, start_2+idx_y*j_2, r)

  
print("=====>Loading image...")
# image = Image.open('./data/input2.jpeg')
image = Image.open('./data/scenergb_3.jpg')
im2 = preprocess2(image)
image = preprocess(image)
image = Variable(image)
image = image.unsqueeze(0)
image = image.cuda()
 
outputs, outputs_shape = model(image)

heatmap = outputs.data.cpu().numpy()
# 285 corresponds to the class index of cat
# you can change it to other class if you are not detecting cat
cat_heatmap = heatmap[6,:]
cat_heatmap = cat_heatmap.reshape(9,14)

#pick the largest response of heatmap but you can change it to a threshold here
(idx_x, idx_y) = np.unravel_index(cat_heatmap.argmax(), cat_heatmap.shape)
(mid_x, mid_y, r) = receptive_field(idx_y, idx_x)
print("idx_x: " + str(idx_x) + "idx_y: " + str(idx_y))
print("mid_x: " + str(mid_x) + " mid_y: " + str(mid_y)+" with size: " + str(r))

r=100


fig1 = plt.figure(1)
ax = fig1.add_subplot(211, aspect='equal')
sns.heatmap(cat_heatmap)

ax = fig1.add_subplot(212, aspect='equal')

im = Variable(im2).data.cpu().numpy()
ax.imshow(np.transpose(im,(1,2,0)))
rect = patches.Rectangle((mid_x-r/2, mid_y-r/2), r, r, linewidth=2, edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.show()


# _, predicted = torch.max(outputs.data, 1)
# predicted = Variable(predicted)
# labels = {int(key):value for (key, value)
#           in requests.get(LABELS_URL).json().items()}
# print(predicted) #285
# print(labels[predicted.data.cpu().numpy()[0]])






