import torch.nn as nn
from torch.autograd import Variable
from torchvision.models as models

vgg11 = models.vgg11(pretrained=True)
vgg11_head = nn.Sequential(*list(vgg11.modules())[:,-7])