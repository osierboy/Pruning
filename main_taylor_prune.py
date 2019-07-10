import dataset
import Net
import TaylorPrune
from xlwt import *
from  xlrd import *

def Taylor_Prune():

    print("-----------Taylor_Prune-----------")

    # CIFAR10
    train_data = dataset.trainloader()
    test_data = dataset.testloader()


    """VGGM"""
    vggm = Net.restore_VGGM().cuda()
    trainer_vggm = TaylorPrune.TrainPrueFineTuner(train_data, test_data, vggm)
    trainer_vggm.test()
    trainer_vggm.prune()


    # 保存剪枝后的VGGM模型
    Net.save_taylor_VGGM(trainer_vggm.model)

    # 记录数据
    taylor_vggm = trainer_vggm.taylorData

    # taylor_vggm.Print()

    write_vggm = Workbook()

    # sheet1记录剪枝过程中train和test精度的变化
    sheet1 = write_vggm.add_sheet('Taylor_train_and_test_acc')
    sheet1.write(1, 0, '剩余filters占比')
    sheet1.write(2, 0, '剪枝后test_acc')
    sheet1.write(3, 0, 'epoch 1后train_acc')
    sheet1.write(4, 0, 'epoch 1后test_acc')
    sheet1.write(5, 0, 'epoch 2后train_acc')
    sheet1.write(6, 0, 'epoch 2后test_acc')

    for i in range(len(taylor_vggm.filter_rate)):
        sheet1.write(0, i + 1, i + 1)
        sheet1.write(1, i + 1, taylor_vggm.filter_rate[i])
        sheet1.write(2, i + 1, taylor_vggm.train_acc_w[i])
        sheet1.write(3, i + 1, taylor_vggm.train_acc1[i])
        sheet1.write(4, i + 1, taylor_vggm.test_acc1[i])
        sheet1.write(5, i + 1, taylor_vggm.train_acc2[i])
        sheet1.write(6, i + 1, taylor_vggm.test_acc2[i])

    # sheet2记录不同剪枝程度时各层剩余的filters数量
    sheet2 = write_vggm.add_sheet('Taylor_layer_prune')
    sheet2.write(1, 0, 'layer_prune_0')
    sheet2.write(2, 0, 'layer_prune_4')
    sheet2.write(3, 0, 'layer_prune_7')
    sheet2.write(4, 0, 'layer_prune_98')

    for i, layer_index in enumerate(taylor_vggm.layer_prune_0):
        sheet2.write(0, i + 1, layer_index)  # 第一行记录层的索引
        sheet2.write(1, i + 1, taylor_vggm.layer_prune_0[layer_index])
        sheet2.write(2, i + 1, taylor_vggm.layer_prune_4[layer_index])
        sheet2.write(3, i + 1, taylor_vggm.layer_prune_7[layer_index])
        sheet2.write(4, i + 1, taylor_vggm.layer_prune_98[layer_index])

        # sheet3记录不同剪枝程度时的top1以及最终的final test acc
    sheet3 = write_vggm.add_sheet('Taylor_top1')
    sheet3.write(1, 0, 'top1_0')
    sheet3.write(1, 1, taylor_vggm.top1_0)
    sheet3.write(2, 0, 'top1_4')
    sheet3.write(2, 1, taylor_vggm.top1_4)
    sheet3.write(3, 0, 'top1_7')
    sheet3.write(3, 1, taylor_vggm.top1_7)
    sheet3.write(4, 0, 'top1_98')
    sheet3.write(4, 1, taylor_vggm.top1_98)
    sheet3.write(5, 0, 'final_test_acc')
    sheet3.write(5, 1, taylor_vggm.final_test_acc)

    write_vggm.save('VGGM_Taylor.xls')



    """VGG16"""
    vgg16 = Net.restore_VGG16().cuda()
    trainer_vgg16 = TaylorPrune.TrainPrueFineTuner(train_data, test_data, vgg16)
    trainer_vgg16.test()
    trainer_vgg16.prune()

    # 保存剪枝后的VGG16模型
    Net.save_taylor_VGG16(trainer_vgg16.model)

    # 记录数据
    taylor_vgg16 = trainer_vgg16.taylorData

    write_vgg16 = Workbook()

    # sheet1记录剪枝过程中train和test精度的变化
    sheet1 = write_vgg16.add_sheet('Taylor_train_and_test_acc')
    sheet1.write(1, 0, '剩余filters占比')
    sheet1.write(2, 0, '剪枝后test_acc')
    sheet1.write(3, 0, 'epoch 1后train_acc')
    sheet1.write(4, 0, 'epoch 1后test_acc')
    sheet1.write(5, 0, 'epoch 2后train_acc')
    sheet1.write(6, 0, 'epoch 2后test_acc')

    for i in range(len(taylor_vgg16.filter_rate)):
        sheet1.write(0, i+1, i+1)
        sheet1.write(1, i+1, taylor_vgg16.filter_rate[i])
        sheet1.write(2, i+1, taylor_vgg16.train_acc_w[i])
        sheet1.write(3, i+1, taylor_vgg16.train_acc1[i])
        sheet1.write(4, i+1, taylor_vgg16.test_acc1[i])
        sheet1.write(5, i+1, taylor_vgg16.train_acc2[i])
        sheet1.write(6, i+1, taylor_vgg16.test_acc2[i])

    # sheet2记录不同剪枝程度时各层剩余的filters数量
    sheet2 = write_vgg16.add_sheet('Taylor_layer_prune')
    sheet2.write(1, 0, 'layer_prune_0')
    sheet2.write(2, 0, 'layer_prune_4')
    sheet2.write(3, 0, 'layer_prune_7')
    sheet2.write(4, 0, 'layer_prune_98')

    for i, layer_index in enumerate(taylor_vgg16.layer_prune_0):
        sheet2.write(0, i+1, layer_index)   # 第一行记录层的索引
        sheet2.write(1, i+1, taylor_vgg16.layer_prune_0[layer_index])
        sheet2.write(2, i+1, taylor_vgg16.layer_prune_4[layer_index])
        sheet2.write(3, i+1, taylor_vgg16.layer_prune_7[layer_index])
        sheet2.write(4, i+1, taylor_vgg16.layer_prune_98[layer_index])

     # sheet3记录不同剪枝程度时的top1以及最终的final test acc
    sheet3 = write_vgg16.add_sheet('Taylor_top1')
    sheet3.write(1, 0, 'top1_0')
    sheet3.write(1, 1, taylor_vgg16.top1_0)
    sheet3.write(2, 0, 'top1_4')
    sheet3.write(2, 1, taylor_vgg16.top1_4)
    sheet3.write(3, 0, 'top1_7')
    sheet3.write(3, 1, taylor_vgg16.top1_7)
    sheet3.write(4, 0, 'top1_98')
    sheet3.write(4, 1, taylor_vgg16.top1_98)
    sheet3.write(5, 0, 'final_test_acc')
    sheet3.write(5, 1, taylor_vgg16.final_test_acc)

    write_vgg16.save('VGG16_Taylor.xls')


    """VGG19"""
    vgg19 = Net.restore_VGG19().cuda()
    trainer_vgg19 = TaylorPrune.TrainPrueFineTuner(train_data, test_data, vgg19)
    trainer_vgg19.test()
    trainer_vgg19.prune()

    # 保存剪枝后的VGG19模型
    Net.save_taylor_VGG19(trainer_vgg19.model)

    # 记录数据
    taylor_vgg19 = trainer_vgg19.taylorData

    write_vgg19 = Workbook()

    # sheet1记录剪枝过程中train和test精度的变化
    sheet1 = write_vgg19.add_sheet('Taylor_train_and_test_acc')
    sheet1.write(1, 0, '剩余filters占比')
    sheet1.write(2, 0, '剪枝后test_acc')
    sheet1.write(3, 0, 'epoch 1后train_acc')
    sheet1.write(4, 0, 'epoch 1后test_acc')
    sheet1.write(5, 0, 'epoch 2后train_acc')
    sheet1.write(6, 0, 'epoch 2后test_acc')

    for i in range(len(taylor_vgg19.filter_rate)):
        sheet1.write(0, i + 1, i + 1)
        sheet1.write(1, i + 1, taylor_vgg19.filter_rate[i])
        sheet1.write(2, i + 1, taylor_vgg19.train_acc_w[i])
        sheet1.write(3, i + 1, taylor_vgg19.train_acc1[i])
        sheet1.write(4, i + 1, taylor_vgg19.test_acc1[i])
        sheet1.write(5, i + 1, taylor_vgg19.train_acc2[i])
        sheet1.write(6, i + 1, taylor_vgg19.test_acc2[i])

    # sheet2记录不同剪枝程度时各层剩余的filters数量
    sheet2 = write_vgg19.add_sheet('Taylor_layer_prune')
    sheet2.write(1, 0, 'layer_prune_0')
    sheet2.write(2, 0, 'layer_prune_4')
    sheet2.write(3, 0, 'layer_prune_7')
    sheet2.write(4, 0, 'layer_prune_98')

    for i, layer_index in enumerate(taylor_vgg19.layer_prune_0):
        sheet2.write(0, i + 1, layer_index)  # 第一行记录层的索引
        sheet2.write(1, i + 1, taylor_vgg19.layer_prune_0[layer_index])
        sheet2.write(2, i + 1, taylor_vgg19.layer_prune_4[layer_index])
        sheet2.write(3, i + 1, taylor_vgg19.layer_prune_7[layer_index])
        sheet2.write(4, i + 1, taylor_vgg19.layer_prune_98[layer_index])

        # sheet3记录不同剪枝程度时的top1以及最终的final test acc
    sheet3 = write_vgg19.add_sheet('Taylor_top1')
    sheet3.write(1, 0, 'top1_0')
    sheet3.write(1, 1, taylor_vgg19.top1_0)
    sheet3.write(2, 0, 'top1_4')
    sheet3.write(2, 1, taylor_vgg19.top1_4)
    sheet3.write(3, 0, 'top1_7')
    sheet3.write(3, 1, taylor_vgg19.top1_7)
    sheet3.write(4, 0, 'top1_98')
    sheet3.write(4, 1, taylor_vgg19.top1_98)
    sheet3.write(5, 0, 'final_test_acc')
    sheet3.write(5, 1, taylor_vgg19.final_test_acc)

    write_vgg19.save('VGG19_Taylor.xls')



