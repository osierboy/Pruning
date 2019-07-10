import torch
import torch.nn as nn
from torchvision import models
import time
import numpy as np

def replace_layers(model, i, indexes, layers ):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def prune_net_conv_layer(model, layer_index, filter_index):
    _, conv = list(model.features._modules.items())[layer_index]  # 需要转换成列表
    next_conv = None # 记录下一个卷积层
    offset = 1

    #找到当前卷积层的下一个卷积层，两层之间的偏移为offset
    while layer_index + offset < len(model.features._modules.items()):
        res = list(model.features._modules.items())[layer_index + offset]
        if isinstance(res[1], nn.modules.conv.Conv2d):
            next_name, next_conv = res
            break
        offset += 1

    """对当前的conv进行处理"""
    """"对该层，减去rank最小的三维filter"""

    new_conv = nn.Conv2d(
        in_channels = conv.in_channels,
        out_channels = conv.out_channels - 1,
        kernel_size = conv.kernel_size,
        stride = conv.stride,
        padding = conv.padding,
        dilation = conv.dilation,
        groups = conv.groups,
        bias = (conv.bias is not None)
    )

    #对新的卷积层赋值, weight是4维的
    old_weights = conv.weight.data.cpu().numpy()  # 转换为numpy类型
    new_weights = new_conv.weight.data.cpu().numpy()

    # 剪掉该层中的第filter_index个filter
    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index :, :, :, :] = old_weights[filter_index+1 :, :, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    if conv.bias is not None:

        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)

        bias[: filter_index] = bias_numpy[: filter_index]
        bias[filter_index :] = bias_numpy [filter_index+1 :]
        new_conv.bias.data = torch.from_numpy(bias).cuda()

    """对下一层进行处理"""
    if not next_conv is None:

        next_new_conv = nn.Conv2d(
            in_channels=next_conv.in_channels - 1,
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            dilation=next_conv.dilation,
            groups=next_conv.groups,
            bias=(next_conv.bias is not None)
        )
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index :, :, :] = old_weights[:, filter_index + 1 :,  :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        if not next_new_conv.bias is None:
            next_new_conv.bias.data = next_conv.bias.data

    # 存在next_conv时，重建feartures
    if not next_conv is None:
        features = nn.Sequential(
            *(replace_layers(model.features, i, [layer_index, layer_index + offset], [new_conv, next_new_conv]) for i, _ in enumerate(model.features))
        )

        del model.features
        del conv

        model.features = features
        model.features[layer_index + 1] = nn.BatchNorm2d(model.features[layer_index].out_channels)

    else:

        model.features = nn.Sequential(
            *(replace_layers(model.features, i, [layer_index], [new_conv]) for i, _ in enumerate(model.features))
        )
        model.features[layer_index + 1] = torch.nn.BatchNorm2d(model.features[layer_index].out_channels)

        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1

        if old_linear_layer is None:
            raise BaseException("No linear lays found in classifier")

        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)

        new_linear_layer = nn.Linear(old_linear_layer.in_features - params_per_input_channel, old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        # 二维，第一维度时batch
        new_weights[:, : filter_index * params_per_input_channel] = old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = old_weights[:, (filter_index+1) * params_per_input_channel :]

        new_linear_layer.bias.data = old_linear_layer.bias.data
        new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

        classifier = nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], [new_linear_layer]) for i,_ in enumerate(model.classifier))
        )

        del model.classifier
        del new_conv
        del conv
        model.classifier = classifier

    return model