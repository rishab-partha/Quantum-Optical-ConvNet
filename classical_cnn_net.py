import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torchvision
import pandas as pd
import numpy as np
import glob
import cv2
import warnings
import mnist

warnings.filterwarnings("ignore")

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.0009
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False

print("Importing data...")

img_frame = pd.read_csv('data.csv')
labels = img_frame['names']
datum = img_frame.loc[:,'col_0':].to_numpy()

print("Shuffling Data...")

np.random.seed(42)
np.random.shuffle(labels)
np.random.seed(42)
np.random.shuffle(datum)

'''
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(torch.from_numpy(datum[i].reshape((28,28))), cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])

plt.show()
'''

print("Training...")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*60000 for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx in range(938):
      if batch_idx < 937:
        data = datum[batch_idx*64:batch_idx*64+64].reshape(64, 1, 28, 28)
        target = labels[batch_idx*64:batch_idx*64+64].values
      else:
        data = datum[59968:60000].reshape(32, 1, 28, 28)
        target = labels[59968:60000].values
      data = torch.from_numpy(data)
      target = torch.from_numpy(target)
    #for batch_idx, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = network(data.float())
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), 60000,
            batch_idx / 9.375, loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx*64) + ((epoch-1)*60000))
        #torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  confusion = np.zeros((10, 10), dtype='i4')
  with torch.no_grad():
    for batch_idx in range(157):
      if batch_idx < 156:
        data = datum[batch_idx*64+60000:batch_idx*64+60064].reshape(64, 1, 28,28)
        target = labels[batch_idx*64+60000:batch_idx*64+60064].values
      else:
        data = datum[69984:70000].reshape(16, 1, 28,28)
        target = labels[69984:70000].values
      data = torch.from_numpy(data)
      target = torch.from_numpy(target)
      #for data, target in test_loader:
      output = network(data.float())
      for i in range(target.shape[0]):
        subsetIndex = np.argmax(output.numpy()[i])
        targetIndex = target[i].numpy()
        confusion[subsetIndex, targetIndex] += 1
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= 10000
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, 10000,
    100. * correct /10000))
  return confusion

network = network.float()

test()
optim_params = {}
optim_params['lr'] = 0.0005
optim_params['momentum'] = .9
confusion = np.zeros((10, 10), dtype='i4')
mnist.train(network, n_epochs = 5, log_interval = 100, optim_params = optim_params, batch_size = 100)
optim_params['lr'] = 0.0001
mnist.train(network, n_epochs = 5, log_interval = 100, optim_params = optim_params, batch_size = 100)
print(mnist.get_acc(network))
torch.save(network.state_dict(), 'model.pth')
print(confusion)

newFile = open('Data/CNN Data.txt', 'w')
newFile.write(str(confusion))

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')

plt.show()

with torch.no_grad():
  output = network(torch.from_numpy(datum[i].reshape((1, 1, 28,28))))

print("Done!")

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(datum[i].reshape((1, 1, 28, 28)), cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()