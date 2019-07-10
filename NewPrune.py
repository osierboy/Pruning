import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from prune import *

import argparse
from operator import itemgetter  # operator模块中的itemgetter用于获取数据的某些维的数据
from heapq import nsmallest

import dataset
import Net

"""
    用于训练与剪枝网络
    1. 训练网络
    2. 剪枝网络
"""

class TrainPrueFineTuner:
    def __init__(self, train_data, test_data, model, lamda):
        self.train_data = train_data
        self.test_data = test_data

        self.model = model
        self.criterion = nn.CrossEntropyLoss() #损失函数采用交叉熵损失
        self.weight_decay = 5e-4

        self.pruner = FilterPrunner(self.model, lamda)

        self.model.train()

        self.running_loss = 0.0

        self.batch_correct = 0.0
        self.batch_total = 0

        self.eight_batch_correct = 0.0
        self.eight_batch_total = 0

        self.taylorData = Net.ExpData()

        self.finetune_acc = 0.0
        self.finemodel = model  # 记录微调过程中性能最好的模型

        self.BN = {}

    def train(self, optimizer = None, epoches = 10, IsPrue = 0): # 默认情况下训练10个epoches
        if optimizer is None:
            # optimizer = optim.SGD(self.model.parameters(), lr = 0.0001, momentum = 0.9, weight_decay=self.weight_decay)  # 采用动量法优化SGD
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=self.weight_decay)

        top1_tra = []
        top1_tes = []
        for i in range(epoches):
            print("Epoch: ", i+1)
            self.running_loss = 0.0
            top1_train = self.train_epoch(optimizer, i, False)
            top1_test = self.test() # 每训练一个epoches就看一下网络当前的精度

            if IsPrue == 1:
                top1_tra.append(top1_train)
                top1_tes.append(top1_test)

            if IsPrue == 2:
                if top1_test > self.finetune_acc:
                    self.finetune_acc = top1_test
                    self.finemodel = self.model


        print("Finish fine truning. ")

        if IsPrue == 1:
            return [top1_tra, top1_tes]


    def train_epoch(self, optimizer = None, epoch = 0, rank_filters = False):  # rank_filters 在剪枝的时候用到
        self.eight_batch_correct = 0.0
        self.eight_batch_total = 0

        self.batch_correct = 0.0
        self.batch_total = 0

        for batch, (inputs, labels) in enumerate(self.train_data):
            # self.train_batch(optimizer, inputs, labels, rank_filters, epoch, batch)
            self.train_batch(optimizer, inputs.cuda(), labels.cuda(), rank_filters, epoch, batch)  # GPU加速

        if rank_filters == False:
            print("The epoch %d accuracy: %d %%" %  (epoch+1, (100 * self.batch_correct / self.batch_total)))
            return 100 * self.batch_correct / self.batch_total


    def train_batch(self, optimizer, inputs, labels, rank_filters, epoch, batch):
        #optimizer.zero_grad()
        self.model.zero_grad()

        if rank_filters:  # 当前为剪枝时
            outputs = self.pruner.forward(inputs) # 计算用用于rank的值
            loss = self.criterion(outputs, labels)
            loss.backward() # 计算完梯度后会自动hook
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            self.running_loss +=loss.item()

            _,pred = torch.max(outputs,1)
            self.eight_batch_total += labels.size(0)
            self.eight_batch_correct += (pred == labels).sum()

            self.batch_total += labels.size(0)
            self.batch_correct += (pred == labels).sum().item()

            if batch % 20 ==0:
                print("[%d %5d] loss: %.3f " % (epoch+1, batch+1, self.running_loss/20))
                self.running_loss = 0.0

                print("20 batch accuracy: %d %%" % (100 * self.eight_batch_correct/self.eight_batch_total))
                self.eight_batch_correct = 0.0
                self.eight_batch_total = 0

    def test(self):
        self.model.eval() # 模型设置为评估模式，模型的参数不可变化

        correct = 0
        total = 0

        for i ,(inputs, labels) in enumerate(self.test_data):
            inputs = inputs.cuda()
            outputs = self.model(inputs)
            _,pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()

        top1 = 100 * float(correct) / total
        print("Accuracy of the network: %d %%" % top1)

        self.model.train() # 模型参数可训练

        return top1

    def prune(self):

        top1 = self.test()  # 测试一下当前较大网络的性能

        self.taylorData.top1_0 = top1  # 未剪枝时的精度

        self.model.train()

        # optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=self.weight_decay)
        # self.train(optimizer, epoches=2)

        for param in self.model.features.parameters():
            param.requires_grad = True

        numbers_of_filters = self.total_filters()

        # 初始化self.taylorData.layer_prune_0 与 self.taylorData.layer_prune
        for layer_index, (name, layer) in enumerate(self.model.features._modules.items()):
            if isinstance(layer, nn.Conv2d):
                self.taylorData.layer_prune_0[layer_index] = layer.out_channels  # 记录没有剪枝时每一层的filters数量
                self.taylorData.layer_prune[layer_index] = 0  # 初始化每一层剪掉的filters个数为0
        print('layer_prune_0: ', self.taylorData.layer_prune_0)

        print("Number of  filters is ", numbers_of_filters)

        num_filters_once_prune = 64

        iterations = int(float(numbers_of_filters) / num_filters_once_prune)

        # 计算不同剪枝比需要的剪枝迭代次数
        rate_iter = []
        rate_iter.append(int(iterations * 40.0 / 100))
        rate_iter.append(int(iterations * 70.0 / 100))
        rate_iter.append(int(iterations * 98.0 / 100))

        iterations = int(iterations * 98.0 /100)

        print("Number of pruning iterations to reduce 98% filters", iterations)

        for _ in range(iterations):

            print("Prune %d :" %  (_+1))

            print("Ranking filters...")

            #得到需要剪枝的filters的位置
            prune_targets = self.get_candidates_to_prune(num_filters_once_prune)

            layers_pruned = {} # 记录每一层卷积层修剪卷积核的数量

            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_pruned:
                    layers_pruned[layer_index] = 0
                layers_pruned[layer_index] += 1

             # 迭代(_+1)次后每一层的剪枝个数
            for layer_index in layers_pruned:
                self.taylorData.layer_prune[layer_index] += layers_pruned[layer_index]

            print("Layers that will be pruned ", layers_pruned)
            print("Prunning filters..")
            model = self.model.cpu()

            for layer_index, filter_index in prune_targets:
                model = prune_net_conv_layer(model, layer_index, filter_index)  # 网络剪枝，逐个filter的剪

            self.model = model.cuda()

            message = str(100 * float(self.total_filters()) / numbers_of_filters) + "%"

            print("Filters pruned ", str(message))

            self.taylorData.filter_rate.append(
                100 * int(float(self.total_filters()) / numbers_of_filters))  # 记录网络剩余filter的占比

            top1_w = self.test()  # 剪枝后没有进行微调的精度
            self.taylorData.train_acc_w.append(top1_w)

            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=self.weight_decay)
            [top1_tra, top1_tes] = self.train(optimizer, epoches=2, IsPrue=1)
            self.taylorData.train_acc1.append(top1_tra[0])
            self.taylorData.train_acc2.append(top1_tra[1])
            self.taylorData.test_acc1.append(top1_tes[0])
            self.taylorData.test_acc2.append(top1_tes[1])

            # 某剪枝比例后每一层剩余的filters个数 以及 top1
            if _ + 1 == rate_iter[0]:
                for layer_index in self.taylorData.layer_prune_0:
                    self.taylorData.layer_prune_4[layer_index] = self.taylorData.layer_prune_0[layer_index] - \
                                                                 self.taylorData.layer_prune[
                                                                     layer_index]  # 剪枝40%后每一层剩余的filters个数
                    self.taylorData.top1_4 = top1_tes[1]
            elif _ + 1 == rate_iter[1]:
                for layer_index in self.taylorData.layer_prune_0:
                    self.taylorData.layer_prune_7[layer_index] = self.taylorData.layer_prune_0[layer_index] - \
                                                                 self.taylorData.layer_prune[
                                                                     layer_index]  # 剪枝70%后每一层剩余的filters个数
                    self.taylorData.top1_7 = top1_tes[1]
            elif _ + 1 == rate_iter[2]:
                for layer_index in self.taylorData.layer_prune_0:
                    self.taylorData.layer_prune_98[layer_index] = self.taylorData.layer_prune_0[layer_index] - \
                                                                  self.taylorData.layer_prune[
                                                                      layer_index]  # 剪枝70%后每一层剩余的filters个数
                    self.taylorData.top1_98 = top1_tes[1]

        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=30, IsPrue=2)
        self.model = self.finemodel  # 更新为表现得最好得模型
        top1 = self.test()
        self.taylorData.final_test_acc = top1


    def total_filters(self):
        filters = 0
        for layer_index, (name, layer) in enumerate(self.model.features._modules.items()):
            if isinstance(layer, nn.Conv2d):
                filters += layer.out_channels
        return filters

    def get_candidates_to_prune(self, num_filters_once_prune):
        self.pruner.reset()  # 初始化filter_ranks

        # 这一步获得了所有filters的重要值存在rank_filters中
        self.train_epoch(rank_filters = True) # 利用train_data 进行一次epoch训练，rank_filter = True时会执行pruner.forward()得到filter_ranks,但是只是计算每个filter的优先级，并不进行更新

        # 获取所有层的BN向量
        index = 0
        for name, layer in self.model.features._modules.items():
            if isinstance(layer, nn.BatchNorm2d):
                self.BN[index] = layer.weight
                index += 1

        self.pruner.getBN(self.BN) # 初始化BN

        self.pruner.norm_every_layer_ranks()  # 对filter_ranks进行正则化
        self.pruner.norm_BN() # 对BN正则化
        self.pruner.get_new_layer_ranks()  # 整合filter_ranks与BN系数

        return self.pruner.get_pruning_plan(num_filters_once_prune)  # 排序得到可以裁剪的对象

"""
    2. 用于对网络进行剪枝
"""

class FilterPrunner:
    def __init__(self,model,lamda):
        self.model = model
        self.lamda = lamda
        self.reset()

    def reset(self):
        self.filter_ranks = {}  # 记录每一个filter的级别
        self.BN = {} # 记录每一个feature map的BN系数

    def getBN(self, BN):
        self.BN = BN

    def forward(self, x):
        self.activations = []  # 记录每一卷积层输出的激励值
        self.activation_index_to_layer = {} # 相应activation中对应layer的索引

        self.gradients = []
        self.grad_index = 0

        activation_index = 0

        for layer_index, (name, layer) in enumerate(self.model.features._modules.items()):
            x = layer(x) # 对该层进行计算
            if isinstance(layer, nn.Conv2d):
                x.register_hook(self.compute_rank) # 对x注册hook，以便得到卷积层中的梯度和激活函数值（每一层的输出都会注册hook）， 一旦梯度计算完成， 便会自动调用compute_rank
                self.activations.append(x) # 记录当前卷积层的激活输出值
                self.activation_index_to_layer[activation_index] = layer_index # 记录对应的层号
                activation_index += 1
        return self.model.classifier(x.view(x.size(0), -1))

    # 钩子注册的函数， activation的维度为 batch filter h w，钩子函数只对对应的一个卷积层进行操作  activations为5维的
    def compute_rank(self, grad):   # 被forward调用，用于grad更新时，计算值用于排序

        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index] # 取出第activation_index 层的输出，维度为4维，batch，filter，h，w

        # 变化1，对 feature map*grad 进行F范式代表重要性
        values = torch.sum((activation * grad)*(activation * grad), dim = 2).sum(dim = 2).data.sqrt()
        # values = torch.sum(values / (activation.size(2) * activation.size(3)), dim = 0)
        values = values.sum(dim = 0)
        values = values/activation.size(0)

        if activation_index not in self.filter_ranks: # 当没对actiovation_index层进行操作时
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1


    def norm_every_layer_ranks(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v).cpu()).cuda()
            self.filter_ranks[i] = v.cpu()

            # print(i, self.filter_ranks[i])

    def norm_BN(self):
        for i in self.BN:
            v = torch.abs(self.BN[i])
            v = v / torch.sqrt(torch.sum(v * v)).cuda()
            self.BN[i] = v.cpu()

    def get_new_layer_ranks(self):
        # print("filter_ranks, ", self.filter_ranks)
        # print("BN, ", self.BN)

        for i in self.filter_ranks:
            self.filter_ranks[i] = self.lamda*self.filter_ranks[i] + (1-self.lamda)*self.BN[i]

        # print("filter_ranks, ", self.filter_ranks)


    def get_pruning_plan(self, num_filter_once_prune):   # 对filter排序，得到可以裁剪的对象，裁剪数量为num_filters_once_prune
        filters_to_prune = self.lowest_rank_filters(num_filter_once_prune) # 找对最低rank的filter，包括对应的层，filter，以及对应的rank

        filters_to_prune_per_layer = {}
        for(l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:  # 哪些层需要剪枝
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)  #将该层需要剪掉的filter加入

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] -= i

        filters_to_prune = [] # 标记需要呗剪掉的filter的位置
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l,i))
        return filters_to_prune

    def lowest_rank_filters(self, num_filter_once_prune):
        data = [] # 3维数组，表示每一个filter的级别, 每一个卷积的卷积层号，卷积在该层的序号，该卷积的rank
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_index_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num_filter_once_prune, data, itemgetter(2)) # 根据data第二维度的大小返回data中num_filter_once_prune个数据