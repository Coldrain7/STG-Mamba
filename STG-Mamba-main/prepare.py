import os

import torch.utils.data as utils
import numpy as np
import torch


def PrepareDataset(speed_matrix, BATCH_SIZE=48, seq_len=12, pred_len=12, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare Train & Test datasets and dataloaders

    Convert traffic/weather/volume matrix to train and test dataset.

    Args:
        speed_matrix: The whole spatial-temporal dataset matrix. (It doesn't necessarily means speed, but can also be flow or weather matrix).
        seq_len: The length of input sequence.
        pred_len: The length of prediction sequence, match the seq_len for model compatibility.
    Return:
        Train_dataloader
        Test_dataloader
    """
    time_len = speed_matrix.shape[0]  # 行的长度，对应于数据表中即为时间数据的长度
    # max_speed = speed_matrix.max().max()
    # speed_matrix = speed_matrix / max_speed

    # MinMax Normalization Method.
    max_speed = speed_matrix.max().max()
    min_speed = speed_matrix.min().min()
    speed_matrix = (speed_matrix - min_speed) / (max_speed - min_speed)  # 归一化

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        speed_sequences.append(speed_matrix.iloc[i:i + seq_len].values)  # 取一个从i开始的长seq_len的序列加入到speed_sequences中
        speed_labels.append(speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values)  # 取上一步序列后的pred_len长度的序列作为预测标签
    speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)  # 将列表转成numpy数组
    print(f'speed_sequences.shape = {speed_sequences.shape}')
    print(f'speed_labels.shape = {speed_labels.shape}')
    # Reshape labels to have the same second dimension as the sequences
    speed_labels = speed_labels.reshape(speed_labels.shape[0], seq_len, -1)

    # shuffle & split the dataset to training and testing sets
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)  # 生成下标数组，类型为ndarray
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))  # floor为向下取整
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))  # 表示验证集的最大索引

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]
    # 将numpy数组转换成Pytorch的tensor以进行训练
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    torch.save((train_data, train_label), 'train_set.pth')
    torch.save((valid_data, valid_label), 'valid_set.pth')
    torch.save((test_data, test_label), 'test_set.pth')

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # drop_last=True时，如果数据集的大小不能被batch_size整除，那么DataLoader将不会返回最后一个不完整的批次。
    return train_dataloader, valid_dataloader, test_dataloader, max_speed
def load_dataset(BATCH_SIZE=48):
    train_data, train_label = torch.load('train_set.pth')
    valid_data, valid_label = torch.load('valid_set.pth')
    test_data, test_label = torch.load('test_set.pth')
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader


def get_dataloader(speed_matrix, BATCH_SIZE=48):
    file_name = 'train_set.pth'
    for root, dirs, files in os.walk(os.getcwd()):
        if file_name in files:
            # 找到文件，返回其完整路径
            print('从文件中加载数据集')
            return load_dataset(BATCH_SIZE)
    train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE)
    return train_dataloader, valid_dataloader, test_dataloader
