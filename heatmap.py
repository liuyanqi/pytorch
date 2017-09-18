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
import numpy as np


sns.set()
use_cuda = torch.cuda.is_available()
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

print("=====>Loading model...")
vgg11 = models.vgg11(pretrained=True)
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
plt.figure(1)
ax = sns.heatmap(cat_heatmap)

plt.figure(2)
im = Variable(im2).data.cpu().numpy()
plt.imshow(np.transpose(im,(1,2,0)))

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




