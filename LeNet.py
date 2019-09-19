import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import argparse


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='test', type=str, help='train or test')
parser.add_argument('--epoch', default=20, type=str, help='total epochs of training')
parser.add_argument('--lr', default=0.1, type=str, help='learning rate of training')
parser.add_argument('--download', default=True, type=bool, help='download mnist data or not. True of False')
parser.add_argument('--check_term', default=100, type=int, help='loss check term.')
parser.add_argument('--optim', default='SGD', type=str, help='optimizer. SGD or ADAM')
parser.add_argument('--momentum', default=0.0, type=float, help='momentum in SGD.')
args = parser.parse_args()

# Lenet-5 module
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2) # by padding=2, makes size of mnist data 28x28 to 32x32.
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1,16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output

# transforms input data to tensor
transforms = transforms.ToTensor()

# check cuda availability
if not torch.cuda.is_available():
    print('Your cuda is False.')
    assert(0)

# model training in train mode
if args.mode == 'train':
    trainset = torchvision.datasets.MNIST(root='./MNIST', train=True, transform=transforms, download=args.download)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
    # define loss function
    criterion=nn.CrossEntropyLoss()
    # define LeNet model, change mode to train and assign model to cuda memory
    model = LeNet_5()
    model.train()
    model.cuda()
    # define optimizer.
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # training
    for ep in range(args.epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('%d epoch training loss : %f' % (ep+1, running_loss / len(trainloader)))
        running_loss = 0.0
    torch.save(model, './checkpoint/checkpoint_%depoch.pth' % (args.epoch))

if args.mode == 'test':
    valset = torchvision.datasets.MNIST(root='./MNIST', train=False, transform=transforms, download=args.download)
    valloader = DataLoader(valset, batch_size=100, shuffle=False)
    model=torch.load('./checkpoint/checkpoint_%depoch.pth' % (args.epoch))
    model.eval()
    model.cuda()
    with torch.no_grad():
        correct = 0
        for j, data in enumerate(valloader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
        print('accuracy : %4f' % (correct/len(valset)))