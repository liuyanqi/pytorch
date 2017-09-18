import torch

print "resnet10"
test_10 = torch.load('./checkpoint/ckpt10.ta')
print "best acc " + str(test_10['acc'])
train_10 = torch.load('./checkpoint/check_train_10')
print "duration " + str(sum(train_10['training_time'])/60)

print "resnet12"
test_12 = torch.load('./checkpoint/ckpt12.ta')
print "best acc " + str(test_12['acc'])
train_12 = torch.load('./checkpoint/check_train_12')
print "duration " + str(sum(train_12['training_time'])/60)

# print "resnet18"
# test_18 = torch.load('./checkpoint/ckpt18.t7')
# print "best acc " + str(test_18['acc'])
# train_18 = torch.load('./checkpoint/check_train_18')
# print "duration " + str(sum(train_18['training_time'])/60)

print "resnet34"
test_34 = torch.load('./checkpoint/ckpt34.ta')
print "best acc " + str(test_34['acc'])
train_34 = torch.load('./checkpoint/check_train_34')
print "duration " + str(sum(train_34['training_time'])/60)

print "resnet50"
test_50 = torch.load('./checkpoint/ckpt50.ta')
print "best acc " + str(test_34['acc'])
train_50 = torch.load('./checkpoint/check_train_50')
print "duration " + str(sum(train_50['training_time'])/60)
