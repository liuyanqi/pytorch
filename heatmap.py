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

sns.set()
use_cuda = torch.cuda.is_available()
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

print("=====>Loading model...")
vgg11 = models.vgg11(pretrained=True)

convnet = [[3,1,1], [2,2,0], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [7,1,0], [1,1,0], [1,1,0]]
layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'pool3','conv5', 'conv6', 'pool4', 'conv7', 'conv8', 'pool5', 'conv9', 'conv10', 'conv11']
imsize = 700


vgg11_head = nn.Sequential(*list(vgg11.children())[:-1])
class Fully_Conv_Vgg(nn.Module):
	def __init__(self):
		super(Fully_Conv_Vgg, self).__init__()

		self.base_model = vgg11_head
		self.conv1 = nn.Conv2d(512,4096, kernel_size=7)
		self.conv2 = nn.Conv2d(4096,4096, kernel_size=1)
		self.conv3 = nn.Conv2d(4096, 1000, kernel_size=1)

	def forward(self, x):
		out = self.base_model(x)
		print(out)
		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)

		return out

model = Fully_Conv_Vgg()
#load the pretrained coefficient
model.base_model.load_state_dict(vgg11_head.state_dict())
model.conv1.load_state_dict({"weight": vgg11.classifier[0].state_dict()["weight"].view(512, 4096,7,7),
	                         "bias":vgg11.classifier[0].state_dict()["bias"]})
model.conv2.load_state_dict({"weight": vgg11.classifier[3].state_dict()["weight"].view(4096, 4096,1,1),
	                         "bias":vgg11.classifier[3].state_dict()["bias"]})
model.conv3.load_state_dict({"weight": vgg11.classifier[6].state_dict()["weight"].view(1000, 4096,1,1),
	                         "bias":vgg11.classifier[6].state_dict()["bias"]})

#model = vgg11

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

preprocess = transforms.Compose([
	#transforms.Scale(256),
    transforms.CenterCrop(700),
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

preprocess2 = transforms.Compose([
	#transforms.Scale(256),
    transforms.CenterCrop(700),
    transforms.ToTensor()
])
#receptive field size - 1 + 2 + 1 + 2*2 + 2 + 2*4 +2*4+ 4+2*8 + 2*8+8 + 2*16+ 2*16+ 16 = 150
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
  currentLayer = [imsize, 1, 1, 0.5]
  printLayer(currentLayer, "input image")
  for i in range(len(convnet)):
    currentLayer = outFromIn(convnet[i], currentLayer)
    layerInfos.append(currentLayer)
    printLayer(currentLayer, layer_names[i])
  print ("------------------------")

  
  n = layerInfos[-1][0]
  j = layerInfos[-1][0]
  r = layerInfos[-1][2]
  start = layerInfos[-1][3]


  return (start+idx_x*j, start+idx_y*j, r)

  
print("=====>Loading image...")
#image = Image.open('./data/input2.jpeg')
image = Image.open('./data/input5.jpg')
im2 = preprocess2(image)
image = preprocess(image)
image = Variable(image)
image = image.unsqueeze(0)
image = image.cuda()
 
outputs = model(image)

heatmap = outputs.data.cpu().numpy()
cat_heatmap = heatmap[:,285,:,:]
cat_heatmap = cat_heatmap.reshape(15,15)

(idx_x, idx_y) = np.unravel_index(cat_heatmap.argmax(), cat_heatmap.shape)
(mid_x, mid_y, r) = receptive_field(idx_x, idx_y)
print("mid_x: " + str(mid_x) + " mid_y: " + str(mid_y)+" with size: " + str(r))




plt.figure(1)
ax = sns.heatmap(cat_heatmap)

fig2 = plt.figure(2)
ax = fig2.add_subplot(111, aspect='equal')

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


#convert intermidiate fc layer to conv layer
#load pre-trained fc layer's weights to new convlutional's kernal
# module.conv.load_state_dict({"weight": model.fc.state_dict()["weight"].view(1000, 2048,1, 1), 
# 	                         "bias": model.fc.state_dict()["bias"]})




