import sys
import numpy as np
import torch.utils.data
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision import datasets
class FirstNet(nn.Module):
    def __init__(self,image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
model = FirstNet(image_size=28*28)
x, y, testx, testy = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
x = np.loadtxt(x)
y = np.loadtxt(y)
testx = np.loadtxt(testx)
testy = np.loadtxt(testy)
valuesNum = x.shape[1]
examplesNum = x.shape[0]
checksNum = testx.shape[0]

# minmax original
minmax = np.zeros((3, valuesNum))
minmax[0] = np.amin(x, axis=0)
minmax[1] = np.amax(x, axis=0)
for j in range(valuesNum):
    minmax[2][j] = minmax[1][j] - minmax[0][j]

for i in range(examplesNum):
    for j in range(valuesNum):
        if minmax[2][j] == 0:
            x[i][j] = 0
        else:
            x[i][j] = (x[i][j] - minmax[0][j]) / minmax[2][j]

for i in range(checksNum):
    for j in range(valuesNum):
        if minmax[2][j] == 0:
            testx[i][j] = 0
        else:
            testx[i][j] = (testx[i][j] - minmax[0][j]) / minmax[2][j]
"""
#minmax new: 0-255
for i in range(examplesNum):
    for j in range(valuesNum):
        x[i][j]=(x[i][j])/255
for i in range(checksNum):
    for j in range(valuesNum):
        testx[i][j]=(testx[i][j])/255

#zscore
meani=np.mean(x,axis=0)
sd=np.std(x,axis=0)
for i in range(examplesNum):
    for j in range(valuesNum):
        x[i][j]=(x[i][j]-meani[j])/sd[j]
meani=np.mean(testx,axis=0)
sd=np.std(testx,axis=0)
for i in range(checksNum):
    for j in range(valuesNum):
        testx[i][j]=(testx[i][j]-meani[j])/sd[j]

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms),
                                           batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms), batch_size=64,
                                          shuffle=True)
"""
x = torch.from_numpy(x)
y = torch.from_numpy(y)
testx = torch.from_numpy(testx)
testy = torch.from_numpy(testy)
train=TensorDataset(x, y)
test=TensorDataset(testx, testy)
c=0