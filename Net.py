import torch
import torch.nn as nn

class ExpData:
    def __init__(self):
        self.filter_rate = [] # 剪枝后剩余filter的占比

        self.train_acc_w = []  # 记录每次迭代剪枝后的未进行后fine-tune 的top1
        self.train_acc1 = []  # 记录每次迭代剪枝后的第一次epoch后的train_acc
        self.test_acc1 = []   # 记录每次迭代剪枝后的第一次epoch后的test_acc
        self.train_acc2 = []  # 记录每次迭代剪枝后的第2次epoch后的train_acc
        self.test_acc2 = []  # 记录每次迭代剪枝后的第2次epoch后的test_acc

        self.layer_prune = {}   # 记录每一层剪掉的filters数量
        self.layer_prune_0 = {} # 未剪枝时每一层的filters数量
        self.layer_prune_4 = {} # 剪枝40%后每一层剩余的filters数量
        self.layer_prune_7 = {}  # 剪枝70%后每一层剩余的filters数量
        self.layer_prune_98 = {}  # 剪枝98%后每一层剩余的filters数量

        # 不同剪枝比剪枝前后2次epoch的top1精度
        self.top1_0 = 0
        self.top1_4 = 0
        self.top1_7 = 0
        self.top1_98 = 0

        self.final_test_acc = 0

    def Print(self):
        print(self.filter_rate)
        print(self.train_acc1)
        print(self.test_acc1)
        print(self.train_acc2)
        print(self.test_acc2)

        print(self.layer_prune)
        print(self.layer_prune_0)
        print(self.layer_prune_4)
        print(self.layer_prune_7)
        print(self.layer_prune_98)

        print(self.top1_0)
        print(self.top1_4)
        print(self.top1_7)
        print(self.top1_98)
        print(self.final_test_acc)

"""restore Baseline Net"""
def restore_VGGM():
    net = torch.load('modifiedvgg.pkl')
    return net

def restore_VGG16():
    net = torch.load('vgg16')
    return net

def restore_VGG19():
    net = torch.load('vgg19')
    return net


""" Taylor """

"""Vggm"""
def save_taylor_VGGM(net):
    torch.save(net, 'taylor_vggm.pkl')

def restore_taylor_VGGM():
    net = torch.load('taylor_vggm.pkl')
    return net


""" 保存VGG16 """
def save_taylor_VGG16(net):
    torch.save(net, 'taylor_vgg16')

def restore_taylor_VGG16():
    net = torch.load('taylor_vgg16')
    return net

""" 保存VGG19 """
def save_taylor_VGG19(net):
    torch.save(net, 'taylor_vgg19')

def restore_taylor_VGG19():
    net = torch.load('taylor_vgg19')
    return net

""" New """

"""Vggm"""

def save_new_VGGM(net, name):
    torch.save(net, name+'new_vggm.pkl')

def restore_new_VGGM(name):
    net = torch.load(name+'new_vggm.pkl')
    return net

""" 保存VGG16 """

def save_new_VGG16(net, name):
    torch.save(net, name+'new_vgg16')

def restore_new_VGG16(name):
    net = torch.load(name+'new_vgg16')
    return net

""" 保存VGG19 """

def save_new_VGG19(net, name):
    torch.save(net, name+'new_vgg19')

def restore_new_VGG19(name):
    net = torch.load(name+'new_vgg19')
    return net

# net = restore_VGGM()
# print(net)