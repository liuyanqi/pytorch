from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import pickle

from custom_dataset import CustomFolder

#import torchvision.models as models
from models import *
from utils import progress_bar
from torch.autograd import Variable
import torchvision.models as models
from collections import defaultdict


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=3, type=int, help='number of epoches')
parser.add_argument('--dr', default=1, type=float, help='data reduction denominator')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225))
])


trainset = CustomFolder(root='./data/umich_shape/train/', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)

testset = CustomFolder(root='./data/umich_shape/test/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=40, shuffle=True, num_workers=2)

print("==> Loading model..")

vgg11 = models.vgg11(pretrained=True)
vgg11_head = nn.Sequential(*list(vgg11.children())[:-1])
# mod = list(net.classifier.children())
# mod.pop()
# mod.append(nn.Linear(4096,16))

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
		out1 = out1.view(out1.size(0), out1.size(1))
		out2 = out2.view(out2.size(0), out2.size(1))


		return out1, out2

net = Two_Branch_Net()
#load the pretrained coefficient
net.base_model.load_state_dict(vgg11_head.state_dict())
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

optimizer = optim.SGD([
							{'params': net.base_model.parameters()},
							{'params': net.classifier.parameters(), 'lr': 0.02},
							{'params': net.classifier_shape.parameters()}
						], lr=0.01, momentum=0.9)

# new_classifier = nn.Sequential(*mod)
# net.classifier = new_classifier

# ignored_params = list(map(id,list(net.classifier.children())[-1].parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
# optimizer = optim.SGD([
#     	{'params': base_params},
#     	{'params': list(net.classifier.children())[-1].parameters(), 'lr':args.lr}
#     	], lr=args.lr, momentum=0.9)

criterion = nn.CrossEntropyLoss()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    	

# Training
def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_shape = 0
    correct_shape =0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % args.dr != 0:
            continue 

        if use_cuda:
            inputs, targets1, targets2 = inputs.cuda(), targets[0].cuda(), targets[1].cuda()
        optimizer.zero_grad()
        inputs, targets1, targets2 = Variable(inputs), Variable(targets1), Variable(targets2)

        outputs1, outputs2 = net(inputs)
        loss1 = criterion(outputs1, targets1)
        loss2 = criterion(outputs2, targets2)
        torch.autograd.backward([loss1, loss2])
        # torch.autograd.backward(loss1)
        optimizer.step()

        # train_loss += loss.data[0]
        _, predicted = torch.max(outputs1.data, 1)
        _, predicted_shape = torch.max(outputs2.data, 1)


        total += targets1.size(0)
        correct += predicted.eq(targets1.data).cpu().sum()

        total_shape += targets2.size(0)
        correct_shape += predicted_shape.eq(targets2.data).cpu().sum()

        print("traning: ", 100*correct/total)
        print("traning shape: ", 100*correct_shape/total_shape)



def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_shape = 0
    correct_shape =0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets1, targets2 = inputs.cuda(), targets[0].cuda(), targets[1].cuda()
        
        inputs, targets1, targets2 = Variable(inputs, volatile=True), Variable(targets1), Variable(targets2)
        outputs1, outputs2 = net(inputs)


        _, predicted = torch.max(outputs1.data, 1)
        _, predicted_shape = torch.max(outputs2.data, 1)
        
        total += targets1.size(0)
        correct += predicted.eq(targets1.data).cpu().sum()

        total_shape += targets2.size(0)
        correct_shape += predicted_shape.eq(targets2.data).cpu().sum()
        
        print("testing: ", 100*correct/total)
        print("testing shape: ", 100*correct_shape/total_shape)

   
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.module if use_cuda else net,
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt_vgg19.ta')
        best_acc = acc


for i in range(args.epoch):
	train()
	test()


print('Saving..')
state = {
    'net': net.module if use_cuda else net,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, './checkpoint/model_vgg11.ta')

# ##save weight
# weight_dict=defaultdict(dict)
# #64*3*3*3
# weight_dict["conv1"]["weight"] = net.module.features[0].state_dict()["weight"].cpu().numpy()
# weight_dict["conv1"]["bias"] = net.module.features[0].state_dict()["bias"].cpu().numpy()

# #128*3*3*3
# weight_dict["conv2"]["weight"] = net.module.features[3].state_dict()["weight"].cpu().numpy()
# weight_dict["conv2"]["bias"] = net.module.features[3].state_dict()["bias"].cpu().numpy()

# #256*3*3*3
# weight_dict["conv3"]["weight"] = net.module.features[6].state_dict()["weight"].cpu().numpy()
# weight_dict["conv3"]["bias"] = net.module.features[6].state_dict()["bias"].cpu().numpy()


# #256*3*3*3
# weight_dict["conv4"]["weight"] = net.module.features[8].state_dict()["weight"].cpu().numpy()
# weight_dict["conv4"]["bias"] = net.module.features[8].state_dict()["bias"].cpu().numpy()

# #512*3*3*3
# weight_dict["conv5"]["weight"] = net.module.features[11].state_dict()["weight"].cpu().numpy()
# weight_dict["conv5"]["bias"] = net.module.features[11].state_dict()["bias"].cpu().numpy()

# #512*3*3*3
# weight_dict["conv6"]["weight"] = net.module.features[13].state_dict()["weight"].cpu().numpy()
# weight_dict["conv6"]["bias"] = net.module.features[13].state_dict()["bias"].cpu().numpy()

# #512*3*3*3
# weight_dict["conv7"]["weight"] = net.module.features[16].state_dict()["weight"].cpu().numpy()
# weight_dict["conv7"]["bias"] = net.module.features[16].state_dict()["bias"].cpu().numpy()

# #512*3*3*3
# weight_dict["conv8"]["weight"] = net.module.features[18].state_dict()["weight"].cpu().numpy()
# weight_dict["conv8"]["bias"] = net.module.features[18].state_dict()["bias"].cpu().numpy()

# #fc layer 25088->4096
# weight_dict["fc1"]["weight"] = net.module.classifier[0].state_dict()["weight"].cpu().numpy()
# weight_dict["fc1"]["bias"] = net.module.classifier[0].state_dict()["bias"].cpu().numpy()

# #fc layer 4096->4096
# weight_dict["fc2"]["weight"] = net.module.classifier[3].state_dict()["weight"].cpu().numpy()
# weight_dict["fc2"]["bias"] = net.module.classifier[3].state_dict()["bias"].cpu().numpy()

# #fc layer 4096->16
# weight_dict["fc3"]["weight"] = net.module.classifier[6].state_dict()["weight"].cpu().numpy()
# weight_dict["fc3"]["bias"] = net.module.classifier[6].state_dict()["bias"].cpu().numpy()

# # torch.save(weight_dict, './checkpoint/trained_model_vgg11.t7')

# with open('./checkpoint/trained_model_vgg11.t7', 'wb') as f:
# 	pickle.dump(weight_dict, f, pickle.HIGHEST_PROTOCOL)
