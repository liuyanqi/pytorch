#!/usr/bin/env
# from __future__ import print_function

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
			nn.Conv2d(4096, 4, kernel_size=1)
			)

	def forward(self, x):
		out = self.base_model(x)
		out1 = self.classifier(out)
		out2 = self.classifier_shape(out)
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

# convnet = [[3,1,1], [2,2,0], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0], [3,1,1], [3,1,1], [2,2,0]]
# layer_names = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'pool3','conv5', 'conv6', 'pool4', 'conv7', 'conv8', 'pool5']

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
# 							 "bias":vgg11.classifier[0].state_dict()["bias"]})
# model.conv2.load_state_dict({"weight": vgg11.classifier[3].state_dict()["weight"].view(4096, 4096,1,1),
# 							 "bias":vgg11.classifier[3].state_dict()["bias"]})
# model.conv3.load_state_dict({"weight": vgg11.classifier[6].state_dict()["weight"].view(1000, 4096,1,1),
# 							 "bias":vgg11.classifier[6].state_dict()["bias"]})

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

def removearray(L, arr):
	for a in arr:
		ind = 0
		for idx in range(len(L)):
			if np.array_equal(L[idx], a):
				L.pop(idx)
				break
	return np.asarray(L)
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

def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))
 

def receptive_field():

  #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
  layerInfos = []
  currentLayer = [imwidth, 1, 1, 0.5]

  for i in range(len(convnet)):
	currentLayer = outFromIn(convnet[i], currentLayer)
	layerInfos.append(currentLayer)
	# printLayer(currentLayer, layer_names[i])

  n = currentLayer[0]
  j = currentLayer[1]
  r = currentLayer[2]
  start = currentLayer[3]

  #return the center coordinate of the receptive field with its corresponding size
  # return (start+idx_x*j, start+idx_y*j, r)
  return(start, j, r)


def UoI(truth_center, predict_center, r):
	boxA = [truth_center[0]-r/2, truth_center[1]-r/2, truth_center[0]+r/2, truth_center[1]+r/2]
	boxB = [predict_center[0]-r/2, predict_center[1]-r/2, predict_center[0]+r/2, predict_center[1]+r/2]

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = (xB - xA + 1)*(yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1)*(boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1)*(boxB[3] - boxB[1] + 1)

	uoi = interArea / (boxAArea + boxBArea - interArea)

	return uoi


(start, j, r) = receptive_field()
print("=====>Loading image...")
image = Image.open('./data/scenergb_4.jpg')

image_array = []
image_array.append(Image.open('./data/scenergb_1.jpg'))
image_array.append(Image.open('./data/scenergb_2.jpg'))
image_array.append(Image.open('./data/scenergb_3.jpg'))
image_array.append(Image.open('./data/scenergb_4.jpg'))
image_array.append(Image.open('./data/scenergb_5.jpg'))
image_array.append(Image.open('./data/scenergb_6.jpg'))
image_array.append(Image.open('./data/scenergb_7.jpg'))
image_array.append(Image.open('./data/scenergb_8.jpg'))
image_array.append(Image.open('./data/scenergb_9.jpg'))
image_array.append(Image.open('./data/scenergb_10.jpg'))

ground_truth_array=[]
ground_truth_array.append([295, 115])
ground_truth_array.append([0, 0])
ground_truth_array.append([356, 178])
ground_truth_array.append([453, 204])
ground_truth_array.append([390, 218])
ground_truth_array.append([0, 0])
ground_truth_array.append([279, 108])
ground_truth_array.append([0, 0])
ground_truth_array.append([230, 125])
ground_truth_array.append([325, 330])

ind=0
precision_list =[]
recall_list = []

# image = image_array[9]
# ground_truth = ground_truth_array[9]

for image in image_array[0:2]:
	ground_truth = ground_truth_array[ind]
	ind +=1
	im2 = preprocess2(image)
	image = preprocess(image)

	image = Variable(image)
	image = image.unsqueeze(0)
	image = image.cuda()
	 
	outputs, outputs_shape = model(image)
	sm = nn.Softmax()
	outputs = sm(outputs)
	heatmap = outputs.data.cpu().numpy()
	heatmap_shape = outputs_shape.data.cpu().numpy()

	heatmap = heatmap[13,:]
	heatmap = heatmap.reshape(9,14)
	heatmap_shape = heatmap_shape.reshape(4,9,14)


	fig1 = plt.figure(1)
	ax = fig1.add_subplot(211, aspect='equal')
	sns.heatmap(heatmap)

	ax = fig1.add_subplot(212, aspect='equal')
	im = Variable(im2).data.cpu().numpy()
	ax.imshow(np.transpose(im,(1,2,0)))

	r = 150

	all_option = np.asarray(np.where(heatmap>=0)).T

	for thres in np.logspace(-9, 0, 1000):
	# for thres in np.linspace(0, 0.01, 100):
		tp_count = 0
		fp_count = 0
		fn_count = 0
		pos = np.asarray(np.where(heatmap>thres)).T
		neg = removearray(list(all_option), pos)
		print thres
		# print len(pos), len(neg)

		for idx in pos:
			mid_x = start + idx[1]*j
			mid_y = start + idx[0]*j
			r = 150
			# print("idx_x: " + str(idx[1]) + "idx_y: " + str(idx[0]))
			# print("mid_x: " + str(mid_x) + " mid_y: " + str(mid_y)+" with size: " + str(r))
			predict_true = [mid_x, mid_y]
			uoi = UoI(ground_truth, predict_true, r)
			if uoi > 0.5:
				tp_count+=1
			else:
				fp_count+=1
			# rect = patches.Rectangle((mid_x-r/2, mid_y-r/2), r, r, linewidth=2, edgecolor='r',facecolor='none')
			# ax.add_patch(rect)

		for idx in neg:
			mid_x = start + idx[1]*j
			mid_y = start + idx[0]*j
			predict_false = [mid_x, mid_y]
			uoi = UoI(ground_truth, predict_false, r)
			if uoi > 0.5:
				fn_count+=1

		recall = tp_count/float(tp_count + fn_count)
		#ground_truth = [0,0] indicates no such object
		if (tp_count==0 or ground_truth ==[0,0]):
			precision = 0
			recall =0
		else:
			precision = tp_count/float(tp_count+fp_count)

		precision_list.append(precision)
		recall_list.append(recall)

		print("precision: " + str(precision))
		print("recall: "  +str(recall))

# print("shape prediction: "+ str(heatmap_shape[:,idx_x, idx_y]))

rect2 = patches.Rectangle((ground_truth[0]-r/2, ground_truth[1]-r/2), r, r, linewidth=2, edgecolor='g',facecolor='none')
ax.add_patch(rect2)
fig2 = plt.figure(2)
plt.scatter(recall_list,precision_list)
plt.show()








