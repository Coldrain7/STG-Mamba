import time
import torch.optim as optim
from STGMamba import *

from torch.autograd import Variable
def laptop_train_mamba(model, train_dataloader, valid_dataloader, A, K=3, num_epochs=1, mamba_features=307):
    # 'mamba_features=184' if we use Knowair dataset.
    # 'mamba_features=307' if we use PEMS04_Dataset Dataste,.
    # 'mamba_features=80' if we use HZ_Metro dataset.
    # 获取第一个批次的输入和标签,
    inputs, labels = next(iter(train_dataloader))  # iter返回迭代器，next返回迭代器的下一个批次
    [batch_size, step_size, fea_size] = inputs.size()  # inputs为一个Tensor对象，这一步将inputs的三个维度赋值给三个变量
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    print(f'feature_size = {A.shape[0]}, d_model = {fea_size}, features = {mamba_features}')
    # 实际打印出的结果：feature_size = 184, d_model = 184, features = 184，三个值都一样
    kfgn_mamba = model
    kfgn_mamba.cuda()  # 似乎不是推荐的代码，以下是更推荐的写法
    # 假设我们有一个可用的CUDA设备
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将模型移动到GPU
    # model.to(device)

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-4  # e后的数字表示10的指数，这里即为10的-4次方
    optimizer = optim.AdamW(kfgn_mamba.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                            amsgrad=False)
    # parameters()是参数列表，两个betas值用于计算梯度的一阶矩估计（类似于动量）和二阶矩估计（类似于RMSprop）的指数衰减率,分别对应Adam优化器中的
    # β 1和β2；eps是一个小的正数，用于防止在更新步长计算时出现除以零的错误。weight_decay是权重衰减系数，用于实现L2正则化。amsgrad是一个布尔值，
    # 表示是否使用AMSGrad变种的Adam优化器。

    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    losses_epoch = []  # Initialize the list for epoch losses

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                # Variable能将变量封装为能够自动求导变量，但是现在已合并到Tensor中
                # inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # 重置模型中所有参数的梯度值,否则反向传播计算梯度后会与之前的梯度值累积
            kfgn_mamba.zero_grad()

            labels = torch.squeeze(labels)  # 删除大小为1的维度
            # 实际上是调用了模型的forward函数
            pred = kfgn_mamba(inputs)  # Updated to use new model

            loss_train = loss_MSE(pred, labels)

            optimizer.zero_grad()
            loss_train.backward()
            # 用优化器更新参数
            optimizer.step()

            losses_train.append(loss_train.data)

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            labels_val = torch.squeeze(labels_val)

            pred = kfgn_mamba(inputs_val)
            loss_valid = loss_MSE(pred, labels_val)
            losses_valid.append(loss_valid.data)

            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                # 打印的数据需要转移到cpu上进行操作
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

        loss_epoch = loss_valid.cpu().data.numpy()
        losses_epoch.append(loss_epoch)
    return kfgn_mamba, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]
