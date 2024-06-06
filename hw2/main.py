import warnings
warnings.filterwarnings("ignore")
import sys

from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import confusion_matrix

from tqdm import tqdm


def train_op(model, loader, optimizer, criterion, epoch=1):
    model.train()
    losses = []
    for _ in range(epoch):  # 多次循环遍历数据集
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # 获取输入
            inputs, labels = data

            # 梯度清零
            optimizer.zero_grad()

            # 正向传播 + 反向传播 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
        losses.append(running_loss/i)
    return losses


def eval_op(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def cal_conf_matrix(model, loader):
    model.eval()
    predict = []
    groundtruth = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            groundtruth += labels.numpy().tolist()
            predict += predicted.numpy().tolist()
    groundtruth = np.array(groundtruth).flatten()
    predict = np.array(predict).flatten()
    cm = confusion_matrix(groundtruth, predict)
    return cm



def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载数据集
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                 download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=transform)
    # print(len(trainset))

    # 数据加载器
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 定义类别标签
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    # 初始化网络和优化器
    criterion = nn.CrossEntropyLoss()   # 交叉熵损失
    optimizer = opt_func(net.parameters(), lr=lr)

    f = open(file, 'w')
    # 保存当前的sys.stdout
    original_stdout = sys.stdout
    sys.stdout = f

    start_time = time.time()
    # 训练网络
    loss_record = []
    acc_record = []
    for _ in tqdm(range(num_epoch), desc=f'Training:'):
        # 训练网络
        loss_record += train_op(net, trainloader, optimizer, criterion)
        # 测试网络
        acc = eval_op(net, testloader)
        acc_record.append(acc)
        print(f'Accuracy: %d %%' % (100 * acc))

    end_time = time.time()
    delta_t = end_time - start_time
    print(f'Finished Training in {delta_t}s')

    print(f'Loss: {loss_record}')
    print(f'Acc: {acc_record}')

    cm = cal_conf_matrix(net, testloader)
    print(f'Confusion_Matrix: {cm}')

    sys.stdout = original_stdout
    f.close()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.set_title('Train Loss')
    # ax1.plot(range(1, num_epoch+1), loss_record)
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.set_title('Accuracy')
    # ax2.plot(range(1, num_epoch + 1), acc_record)
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Acc')
    # plt.show()


if __name__ == '__main__':
    file = './result/output_mlp_1.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.1
    main()

    file = './result/output_mlp_2.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.01
    main()

    file = './result/output_mlp_3.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.001
    main()

    file = './result/output_mlp_4.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.01
    main()

    file = './result/output_mlp_5.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.001
    main()

    file = './result/output_mlp_6.txt'
    net = MLP()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.0001
    main()

    file = './result/output_conv_1.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.1
    main()

    file = './result/output_conv_2.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.01
    main()

    file = './result/output_conv_3.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.SGD
    lr = 0.001
    main()

    file = './result/output_conv_4.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.01
    main()

    file = './result/output_conv_5.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.001
    main()

    file = './result/output_conv_6.txt'
    net = ConvModel()
    num_epoch = 30
    batch_size = 64
    opt_func = optim.Adam
    lr = 0.0001
    main()

