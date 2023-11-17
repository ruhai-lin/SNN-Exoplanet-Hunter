# imports
import snntorch as snn
from snntorch import spikeplot as splt, surrogate
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from imblearn.over_sampling import SMOTE


import matplotlib.pyplot as plt  # 可以尝试seaborn
import numpy as np
import itertools
import pandas as pd
import os

# SETTING UP THE STATIC MNIST DATASET ----------------------------------------------------------------------------------------------------------------------------

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# 步骤1: 准备数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.labels = data.iloc[:, 0].values
        self.features = data.iloc[:, 1:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        feature = self.features[idx]

        if self.transform:
            feature = self.transform(feature)

        sample = {'feature': feature, 'label': label - 1}
        return sample


# 加载训练和测试数据集
train_dataset = CustomDataset('exotrain.csv')
test_dataset = CustomDataset('exotest.csv')

# 步骤2: 使用SMOTE来处理不平衡数据
smote = SMOTE(sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(train_dataset.features, train_dataset.labels)

# resampled_train_dataset = CustomDataset(X_resampled, y_resampled)
train_dataset.features = X_resampled
train_dataset.labels = y_resampled

# 步骤3: 创建数据加载器
batch_size = 64
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# 步骤4: 定义神经网络模型
# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Linear(3197, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Linear(64, 64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(32, 2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.softmax = nn.Softmax()

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        cur1 = F.max_pool1d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool1d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        # spk3, mem3 = self.lif3(cur3, mem3)

        # return spk3
        return self.softmax(cur3)

model = Net()

# 步骤5: 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于分类问题 look up binarycross entropy
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 步骤6: 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for data in train_dataloader:
        inputs, labels = data['feature'].float(), data['label']  # 数据类型转换为Float
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(torch.mean(outputs, 0))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0

        for data in test_dataloader:
            inputs, labels = data['feature'].float(), data['label']  # 数据类型转换为Float
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{num_epochs}] Test Loss: {test_loss / len(test_dataloader)} Test Accuracy: {accuracy}%')

# # 训练完成后，您可以保存模型，用于后续推断任务
# torch.save(model.state_dict(), 'custom_model.pth')

#
# # TRAINING THE SNN ----------------------------------------------------------------------------------------------------------------------------------------------
#
# # pass data into the network, sum the spikes over time
# # and compare the neuron with the highest number of spikes
# # with the target
#
# def print_batch_accuracy(data, targets, train=False):
#     output, _ = net(data.view(batch_size, -1))
#     _, idx = output.sum(dim=0).max(1)
#     acc = np.mean((targets == idx).detach().cpu().numpy())
#
#     if train:
#         print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
#     else:
#         print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
#
# def train_printer():
#     print(f"Epoch {epoch}, Iteration {iter_counter}")
#     print(f"Train Set Loss: {loss_hist[counter]:.2f}")
#     print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
#     print_batch_accuracy(data, targets, train=True)
#     print_batch_accuracy(test_data, test_targets, train=False)
#     print("\n")
#
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
# data, targets = next(iter(train_loader))
# data = data.to(device)
# targets = targets.to(device)
# spk_rec, mem_rec = net(data.view(batch_size, -1))
#
# # initialize the total loss value
# loss_val = torch.zeros((1), dtype=dtype, device=device)
#
# # sum loss at every step
# for step in range(num_steps):
#   loss_val += loss(mem_rec[step], targets)
#
# # clear previously stored gradients
# optimizer.zero_grad()
#
# # calculate the gradients
# loss_val.backward()
#
# # weight update
# optimizer.step()
#
# # calculate new network outputs using the same data
# spk_rec, mem_rec = net(data.view(batch_size, -1))
#
# # initialize the total loss value
# loss_val = torch.zeros((1), dtype=dtype, device=device)
#
# # sum loss at every step
# for step in range(num_steps):
#   loss_val += loss(mem_rec[step], targets)
#
# num_epochs = 1
# loss_hist = []
# test_loss_hist = []
# counter = 0
#
# # Outer training loop
# for epoch in range(num_epochs):
#     iter_counter = 0
#     train_batch = iter(train_loader)
#
#     # Minibatch training loop
#     for data, targets in train_batch:
#         data = data.to(device)
#         targets = targets.to(device)
#
#         # forward pass
#         net.train()
#         spk_rec, mem_rec = net(data.view(batch_size, -1))
#
#         # initialize the loss & sum over time
#         loss_val = torch.zeros((1), dtype=dtype, device=device)
#         for step in range(num_steps):
#             loss_val += loss(mem_rec[step], targets)
#
#         # Gradient calculation + weight update
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()
#
#         # Store loss history for future plotting
#         loss_hist.append(loss_val.item())
#
#         # Test set
#         with torch.no_grad():
#             net.eval()
#             test_data, test_targets = next(iter(test_loader))
#             test_data = test_data.to(device)
#             test_targets = test_targets.to(device)
#
#             # Test set forward pass
#             test_spk, test_mem = net(test_data.view(batch_size, -1))
#
#             # Test set loss
#             test_loss = torch.zeros((1), dtype=dtype, device=device)
#             for step in range(num_steps):
#                 test_loss += loss(test_mem[step], test_targets)
#             test_loss_hist.append(test_loss.item())
#
#             # Print train/test loss/accuracy
#             if counter % 50 == 0:
#                 train_printer()
#             counter += 1
#             iter_counter +=1
#
# # RESULT ----------------------------------------------------------------------------------------------------------------------------------------------
#
# # Plot Loss
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# plt.plot(loss_hist)
# plt.plot(test_loss_hist)
# plt.title("Loss Curves")
# plt.legend(["Train Loss", "Test Loss"])
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()


# total = 0
# correct = 0
#
# # drop_last switched to False to keep all samples
# test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)
#
# with torch.no_grad():
#   net.eval()
#   for data, targets in test_loader:
#     data = data.to(device)
#     targets = targets.to(device)
#
#     # forward pass
#     test_spk, _ = net(data.view(data.size(0), -1))
#
#     # calculate total accuracy
#     _, predicted = test_spk.sum(dim=0).max(1)
#     total += targets.size(0)
#     correct += (predicted == targets).sum().item()
