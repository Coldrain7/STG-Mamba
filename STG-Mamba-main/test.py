import torch

from prepare import *
from train_STGmamba import *


def know_air_test():
    speed_matrix = pd.read_csv(
        'D:\pychramProjects\STG-Mamba\STG-Mamba-main\Know_Air\knowair_temperature.csv\knowair_temperature.csv', sep=',')
    STGmamba = torch.load('model/STGmamba_model.pth')
    train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=6)
    print("\nTesting STGmamba model...")
    results = TestSTG_Mamba(STGmamba, test_dataloader, max_value)  # max_value为矩阵中最大的一个值


def linear_test():
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([[1., 1., 1.],
                                             [2., 2., 2.],
                                             [3., 3., 3.],
                                             [4., 4., 4.]])
    linear_layer.bias.data.fill_(0.5)
    output = linear_layer(inputs)
    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(output, output.shape)


def sum_test():
    a = torch.randn(4, 3)
    print(a)
    b = torch.sum(a, 0)
    print(b)


def mae_test():
    prediction = torch.Tensor([4, 6, 8, 12])
    labels = torch.Tensor([2, 4, 6, 8])
    mae = torch.mean(torch.abs(prediction - labels))
    print(mae.item())


def pems04_test():
    print("\nLoading PEMS04_Dataset data...")
    speed_matrix = pd.read_csv('D:\pychramProjects\STG-Mamba\STG-Mamba-main\PEMS04_Dataset\pems04_flow.csv', sep=',')
    A = np.load('D:\pychramProjects\STG-Mamba\STG-Mamba-main\PEMS04_Dataset\pems04_adj.npy')
    train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=6)
    model = torch.load('model/STGmamba_model.pth')
    print("\nTesting STGmamba model...")
    results = TestSTG_Mamba(model, test_dataloader, max_value)  # max_value为矩阵中最大的一个值


if __name__ == "__main__":
    print('choose you option:')
    option = input('1: test model with knowAir;\n2: linear model test；\n3：sum test \n4:mae_test:\n')
    if option == '1':
        know_air_test()
    elif option == '2':
        linear_test()
    elif option == '3':
        sum_test()
    elif option == '4':
        mae_test()
    elif option == '5':
        pems04_test()
    else:
        a = torch.Tensor([[1, 2, 3]])
        b = a.T
        print(f'a.shape:{a.shape}')
        print(f'b:{b}')
        print(f'a @ b:{a @ b}')
        print(f'a * b:{a * b}')
