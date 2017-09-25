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

#import torchvision.models as models
from models import *
from utils import progress_bar
from torch.autograd import Variable
import torchvision.models as models



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--epoch', default=10, type=int, help='number of epoches')
parser.add_argument('--dr', default=1, type=float, help='data reduction denominator')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.484, 0.456, 0.406), (0.229, 0.224, 0.225))
])


trainset = torchvision.datasets.ImageFolder(root='./data/umich/train/', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./data/umich/test/', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=40, shuffle=True, num_workers=2)

print("==> Loading model..")
net = models.vgg11(pretrained=True)
mod = list(net.classifier.children())
mod.pop()
mod.append(nn.Linear(4096,16))
new_classifier = nn.Sequential(*mod)
net.classifier = new_classifier

ignored_params = list(map(id,list(net.classifier.children())[-1].parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
    	{'params': base_params},
    	{'params': list(net.classifier.children())[-1].parameters(), 'lr':args.lr}
    	], lr=args.lr, momentum=0.9)

criterion = nn.CrossEntropyLoss()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    	
    	

# Training
def train():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx % args.dr != 0:
            continue 

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        # print("target:", str(targets.data.cpu()))
        # print("output: ", str(outputs.data.cpu()))
        # print("predicted: ", str(Variable(predicted).data.cpu()))

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print("traning: ", 100*correct/total)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        print("test: ", str(100*correct/total))
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
   
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
