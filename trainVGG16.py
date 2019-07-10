import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import visdom
import dataset
import VGGNet

class Train:
    def __init__(self, train_data, test_data, model):
        self.train_data = train_data
        self.test_data = test_data

        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.model.train() # 网络设置为train模式
        self.lr = 0.001
        self.weight_decay = 5e-4

        self.test_acc = 0.0

        self.viz = visdom.Visdom()

    def test(self):
        self.model.eval() # 模型设置为评估模式，模型的参数不可变化

        with torch.no_grad():
            correct = 0
            total = 0
            running_loss = 0.0

            I = 0
            for i ,(inputs, labels) in enumerate(self.test_data):
                I = i
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.cuda())
                running_loss += loss.item()

                _,pred = torch.max(outputs,1)
                total += labels.size(0)
                correct += (pred.cpu() == labels).sum()

        print("Accuracy of the network: %d %%" % (100 * float(correct)/ total))

        self.model.train() # 模型参数可训练
        return [float(correct)/ total, running_loss/I]  # 每张验证图像的分类精度以及每个batch的平均loss

    def train(self, epoches = 10):
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay) # L2
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # 自动调整学习率

        loss_win = self.viz.line(np.arange(10))
        acc_win = self.viz.line(X=np.column_stack((np.array(0), np.array(0))),
                                Y=np.column_stack((np.array(0), np.array(0))))

        for i in range(epoches):
            print("Epoch: ", i+1)

            # train过程
            running_loss = 0.0
            train_correct = 0.0
            train_total = 0
            B = 0
            for batch, (inputs, labels) in enumerate(self.train_data):
                B = batch
                self.model.zero_grad()

                outputs =self.model(inputs.cuda())
                loss = self.criterion(outputs, labels.cuda())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (pred.cpu() == labels).sum()

            train_loss = running_loss / B
            print("Epoch %d each batch avg loss: %.3f " % (i + 1, train_loss))

            train_acc = float(train_correct) / train_total
            print("Epoch %d avg train accuracy: %d %%" % (i + 1, 100 * train_acc))

            running_loss = 0.0
            train_correct = 0.0
            train_total = 0

            # 验证过程
            test_acc, test_loss =self.test()
            scheduler.step(test_loss)

            # 可视化

            if i == 0:
                self.viz.line(Y=np.array([train_loss]), X=np.array([i + 1]), update='replace', win=loss_win)
                self.viz.line(Y=np.column_stack((np.array([train_acc]), np.array([test_acc]))),
                              X=np.column_stack((np.array([i + 1]), np.array([i + 1]))),
                              win=acc_win, update='replace',
                              opts=dict(legned=['Train_acc', 'Val_acc']))
            else:
                self.viz.line(Y=np.array([train_loss]), X=np.array([i + 1]), update='append', win=loss_win)
                self.viz.line(Y=np.column_stack((np.array([train_acc]), np.array([test_acc]))),
                              X=np.column_stack((np.array([i + 1]), np.array([i + 1]))),
                              win=acc_win, update='append')

            if test_acc > self.test_acc:
                self.test_acc = test_acc
                VGGNet.save_VGG16(self.model)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = VGGNet.VGGNet16().to(device)


    print(net)

    train_data = dataset.trainloader()
    test_data = dataset.testloader()

    trainer = Train(train_data,test_data,net)

    trainer.train(epoches=100)


