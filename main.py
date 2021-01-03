import sys

# from multiprocessing import reduction
import numpy as np
import torch.utils.data
from torch import optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F


class FirstNetC(nn.Module):
    def __init__(self, image_size):
        super(FirstNetC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        d = nn.Dropout(p=0.2)
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = d(x)
        x = F.relu(self.fc1(x))
        x = d(x)
        x = F.relu(self.fc2(x))
        x = d(x)
        return F.log_softmax(x, dim=1)

"""
class FirstNetD(nn.Module):
    def __init__(self, image_size):
        super(FirstNetD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        d = nn.BatchNorm1d(100)
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = d(x)
        x = F.relu(self.fc1(x))
        x = d(x)
        x = F.relu(self.fc2(x))
        x = d(x)
        return F.log_softmax(x, dim=1)
"""

class FirstNet(nn.Module):
    def __init__(self, image_size):
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
        return F.log_softmax(x, dim=1)


class FirstNetE(nn.Module):
    def __init__(self, image_size):
        super(FirstNetE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(x, dim=1)


class FirstNetF(nn.Module):
    def __init__(self, image_size):
        super(FirstNetF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return F.log_softmax(x, dim=1)
def train(epoch, model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(epoch, model,optimizer,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


def main():
    from torchvision import transforms
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=False,
                                                              transform=transforms), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms),
                                              batch_size=64, shuffle=True)
    model = FirstNet(image_size=28 * 28)
    modelC = FirstNetC(image_size=28 * 28)
    # modelD = FirstNetD(image_size=28 * 28)
    modelE = FirstNetE(image_size=28 * 28)
    modelF = FirstNetF(image_size=28 * 28)
    optimizerA = optim.SGD(model.parameters(), lr=0.01)
    optimizerB = optim.Adam(model.parameters(), lr=0.01)

    print("model A:")
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizerA, train_loader)
        test(epoch, model,optimizerA,test_loader)
    print("model B:")
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizerB, train_loader)
        test(epoch, model,optimizerB,test_loader)
    print("model C:")
    for epoch in range(1, 10 + 1):
        train(epoch, modelC, optimizerA, train_loader)
        test(epoch, modelC, optimizerA, test_loader)
    print("model D:")
    """
    for epoch in range(1, 10 + 1):
        train(epoch, modelD, optimizerA, train_loader)
        test(epoch, modelD, optimizerA, test_loader)
    """
    print("model E:")
    for epoch in range(1, 10 + 1):
        train(epoch, modelE, optimizerA, train_loader)
        test(epoch, modelE, optimizerA, test_loader)
    print("model F:")

    for epoch in range(1, 10 + 1):
        train(epoch, modelF, optimizerA, train_loader)
        test(epoch, modelF, optimizerA, test_loader)
    """
    
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
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    testx = torch.from_numpy(testx)
    testy = torch.from_numpy(testy)
    train=TensorDataset(x, y)
    test=TensorDataset(testx, testy)
    train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    """

if __name__ == "__main__":
    main()