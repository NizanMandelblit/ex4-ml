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
import matplotlib.pyplot as plt
trainBach = 8
testBach = 5

class FirstNetC(nn.Module):
    def __init__(self, image_size):
        super(FirstNetC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        d = nn.Dropout(0.2)
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = d(x)
        x = F.relu(self.fc1(x))
        x = d(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FirstNetD(nn.Module):
    def __init__(self, image_size):
        super(FirstNetD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        b1 = nn.BatchNorm1d(100)
        b2 = nn.BatchNorm1d(50)
        x = x.view(-1, self.image_size)
        x = F.relu(b1(self.fc0(x)))
        x = F.relu(b2(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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
        x = self.fc2(x)
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
        x = self.fc5(x)
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
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


def viewing(epochs, lossm, accurm):
    plt.plot(epochs, lossm)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.show()
    plt.plot(epochs, accurm)
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.show()


def train(epoch, model, optimizer, train_loader):
    model.train()
    test_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        """
        if batch_idx >= (40000 / trainBach):
            break
        """
        optimizer.zero_grad()
        output = model(data)
        #for testing
        test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        #end testing
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    test_loss /= len(train_loader.dataset)
    #print(epoch)
    #print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))


def test(epoch, model,optimizer,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= (10000 / testBach):
                break
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print(epoch)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    return test_loss, (100. * correct / len(test_loader.dataset))


def finaltrain(epoch, model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def runmodels(train_loader,test_loader):
    model = FirstNet(image_size=28 * 28)
    modelC = FirstNetC(image_size=28 * 28)
    modelD = FirstNetD(image_size=28 * 28)
    modelE = FirstNetE(image_size=28 * 28)
    modelF = FirstNetF(image_size=28 * 28)

    optimizerA = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs=[]
    lossm=[]
    accurm=[]

    print("model A:")
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizerA, train_loader)
        loss,accur=test(epoch, model, optimizerA, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)
    epochs = []
    lossm = []
    accurm = []

    model = FirstNet(image_size=28 * 28)
    optimizerB = optim.Adam(model.parameters(), lr=0.001)
    print("model B:")
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizerB, train_loader)
        loss,accur=test(epoch, model,optimizerB, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)
    epochs = []
    lossm = []
    accurm = []
    optimizerC = optim.SGD(modelC.parameters(), lr=0.001, momentum=0.9)
    print("model C:")
    for epoch in range(1, 10 + 1):
        train(epoch, modelC, optimizerC, train_loader)
        loss,accur=test(epoch, modelC, optimizerC, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)
    epochs = []
    lossm = []
    accurm = []
    optimizerD = optim.SGD(modelD.parameters(), lr=0.001, momentum=0.9)
    print("model D:")

    for epoch in range(1, 10 + 1):
        train(epoch, modelD, optimizerD, train_loader)
        loss,accur=test(epoch, modelD, optimizerD, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)
    epochs = []
    lossm = []
    accurm = []
    optimizerE = optim.SGD(modelE.parameters(), lr=0.001, momentum=0.9)
    print("model E:")
    for epoch in range(1, 10 + 1):
        train(epoch, modelE, optimizerE, train_loader)
        loss,accur=test(epoch, modelE, optimizerE, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)

    epochs = []
    lossm = []
    accurm = []

    print("model F:")
    optimizerF = optim.Adam(modelF.parameters(), lr=0.001)
    for epoch in range(1, 10 + 1):
        train(epoch, modelF, optimizerF, train_loader)
        loss,accur=test(epoch, modelF, optimizerF, test_loader)
        epochs+=[epoch]
        lossm+=[loss]
        accurm+=[accur]
    viewing(epochs, lossm, accurm)


def finaltest(epoch, model,optimizer,test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            print(pred)


def main():

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    """
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=False,
                                                              transform=transform), batch_size=trainBach, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform),batch_size=testBach, shuffle=True)
    runmodels(train_loader, test_loader)
    
    my_x = np.loadtxt(sys.argv[3])
    my_y = np.loadtxt(sys.argv[4])
    tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    tensor_y = torch.Tensor(my_y)
    tensor_y = tensor_y.long()
    my_dataset = TensorDataset(tensor_x, tensor_y)
    """
    my_x = np.loadtxt(sys.argv[1])
    my_y = np.loadtxt(sys.argv[2])
    tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    tensor_y = torch.Tensor(my_y)
    tensor_y = tensor_y.long()
    my_dataset2 = TensorDataset(tensor_x, tensor_y)
    train_loader = torch.utils.data.DataLoader(my_dataset2, batch_size=trainBach, shuffle=False)
    #test_loader = torch.utils.data.DataLoader(my_dataset,batch_size=testBach, shuffle=False)
    #runmodels(train_loader,test_loader)
    #test_y = np.loadtxt(sys.argv[4])
    result=open(f"test_y","w")
    test_x = np.loadtxt(sys.argv[3])
    test_x = transform(test_x)
    test_x = test_x.float()
    model = FirstNetD(image_size=28 * 28)
    optimizerA = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1, 10 + 1):
        train(epoch, model, optimizerA, train_loader)
    model.eval()
    with torch.no_grad():
        for data in test_x:
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            for p in pred:
                #print(p.item())
                result.write(str(p.item()) + "\n")
            

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