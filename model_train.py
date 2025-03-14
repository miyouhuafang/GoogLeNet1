import copy
import time

import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import LeNet,Inception
import torch.nn as nn


def train_val_data_process():
    root_train_path =r'D:\pycharm\GoodLeNet\cat and dog\data\cats_and_dogs\train'

    normalize = transforms.Normalize(mean=[0.161, 0.149, 0.137], std=[0.057, 0.051, 0.047])
    #定义数据集处理方法变量
    train_transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),normalize])

    #加载dataset
    train_data=ImageFolder(root=root_train_path,transform=train_transform)
    #print(train_data.class_to_idx)


    train_data,val_data=Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])
    train_loader=Data.DataLoader(dataset=train_data,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=2)
    val_loader=Data.DataLoader(dataset=val_data,
                               batch_size=32,
                               shuffle=True,
                               num_workers=2)
    return train_loader,val_loader

def train_model_process(model,train_loader,val_loader,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    #最高准确度
    best_acc = 0.0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        #初始化参数
        train_loss_func = 0.0
        train_acc = 0.0
        val_loss_func = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0

        #对每一个mini-batch训练和计算
        for step,(b_x,b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()
            #面向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            #查找每一行中最大值对应的行标（index）
            pred = torch.argmax(output, dim=1)
            #计算每一个batch的损失函数
            loss = criterion(output, b_y)

            #梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()
            #根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            train_loss_func += loss.item()*b_x.size(0)
            train_acc += torch.sum(pred == b_y.data).item()
            train_num += b_x.size(0)

        #验证循环，在模型训练过程中评估模型性能
        for step,(b_x,b_y) in enumerate(val_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #评估模式
            model.eval()
            output = model(b_x)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss_func += loss.item()*b_x.size(0)
            val_acc += torch.sum(pred == b_y.data).item()
            val_num += b_x.size(0)

        #计算并保存每一次迭代的loss值和准确率
        train_loss_list.append(train_loss_func/train_num)
        val_loss_list.append(val_loss_func/val_num)
        train_acc_list.append(train_acc/train_num)
        val_acc_list.append(val_acc/val_num)

        print("epoch:{} train loss:{:.4f} train acc:{:.4f}".format(epoch,train_loss_list[-1],train_acc_list[-1]))
        print("epoch:{} val loss:{:.4f} val acc:{:.4f}".format(epoch,val_loss_list[-1],val_acc_list[-1]))

        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            #保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        #计算训练和验证的耗时
        time_use = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_use//60, time_use%60))

    #选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, r'D:\pycharm\GoodLeNet\best_model.pth')

    train_process = pd.DataFrame(data={'epoch':range(num_epochs),
                                       'train_loss':train_loss_list,
                                       'train_acc':train_acc_list,
                                       'val_loss':val_loss_list,
                                       'val_acc':val_acc_list})
    return train_process

def matplot_acc_loss(train_process):
    #显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'],train_process['train_loss'],"ro-",label="Train loss")
    plt.plot(train_process['epoch'], train_process['val_loss'], "bs-", label="Val loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'],train_process['train_acc'],"ro-",label="Train accuracy")
    plt.plot(train_process['epoch'], train_process['val_acc'], "bs-", label="Val accuracy")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    #加载model
    GooLeNet_model = LeNet(Inception)

    train_data,val_data = train_val_data_process()
    train_process=train_model_process(GooLeNet_model,train_data,val_data,num_epochs=20)
    matplot_acc_loss(train_process)





