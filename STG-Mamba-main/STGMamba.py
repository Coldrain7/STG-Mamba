import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import DynamicFilterGNN

from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union


# KFGN (Kalman Filtering Graph Neural Networks) Model
class KFGN(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True):
        super(KFGN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K  # K=3
        self.A_list = []
        # A是一个邻接矩阵
        # torch.sum(A,0)把A沿第0轴的数全部加起来，得到一个1 * feature_size大小的矩阵
        # torch.diag()则会将一维向量作为对角线构造对角矩阵，其余元素全为0
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        # 矩阵乘法，对邻接矩阵进行度归一化
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A
        # 生成维数为feature_size的单位矩阵
        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                # 让A中的元素最大为1
                A_temp = torch.clamp(A_temp, max=1.)
            # A_list中装入度归一化后的邻接矩阵与单位矩阵相乘的结果
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList(
            # gc_list中包含K个DynamicFilterGNN，区别在于构建时取不同的A_list中的矩阵
            [DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.il = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        # 创建一个feature_size长度的一维张量
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        # 重新设置张量中的值
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        input_size = self.feature_size

        self.rfl = nn.Linear(input_size + hidden_size, hidden_size)
        self.ril = nn.Linear(input_size + hidden_size, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size, hidden_size)

        # addtional vars
        self.c = torch.nn.Parameter(torch.Tensor([1]))

        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 64)
        self.fc5 = nn.Linear(64, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, 64)

    def forward(self, input, Hidden_State=None, Cell_State=None, rHidden_State=None, rCell_State=None):
        batch_size, time_steps, _ = input.shape
        if Hidden_State is None:
            Hidden_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if rHidden_State is None:
            rHidden_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if rCell_State is None:
            rCell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        # 将这四个张量增加一维并将序号为1的维度增加为time_steps
        Hidden_State = Hidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)
        rHidden_State = rHidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        rCell_State = rCell_State.unsqueeze(1).expand(-1, time_steps, -1)
        x = input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            # 沿着第一维拼接gc_list中的模型输出
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        dim1 = combined.shape[0]
        dim2 = combined.shape[1]
        dim3 = combined.shape[2]
        # 修改combined的维度
        combined = combined.view(dim1, dim2 // 4, dim3 * 4)
        # 线性层输入三维张量[batch_size, n, in_features]，经过线性层，可以得到输出张量[batch_size, n, out_features].
        # combined沿第1维拼了4次，经过view之后第1维变为原来的长度，第3维变为原来的4倍
        # 第3维变为4倍对应于全连接层的(k+1)*feature_size的输入
        # sigmoid与tanh会计算出张量每个元素的函数值
        # ===========================================================
        # 计算Hidden_State与之后计算rHidden_State的步骤基本上是一个LSTM模块
        # ===========================================================
        f = torch.sigmoid(self.fl(combined))  # 线性层也相当于一个权重矩阵
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        # Cell_State的shape为(batch_size, step, feature_size), 可以与长度为feature_size的向量进行点积计算
        # torch.mul进行计算的两个矩阵从后向前的维度需要相同
        # NC计算出来为(batch_size, step, feature_size)矩阵
        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        # tensor之间用*表示进行Hadamard乘积
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # LSTM
        rcombined = torch.cat((input, rHidden_State), 1)
        d1 = rcombined.shape[0]
        d2 = rcombined.shape[1]
        d3 = rcombined.shape[2]
        # rcombined只经过一次拼接得到，所以这里数字是2
        rcombined = rcombined.view(d1, d2 // 2, d3 * 2)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        # torch.var实际为计算张量所有元素的样本方差,
        var1, var2 = torch.var(input), torch.var(gc)
        # tensor与一个数进行/运算，实际为tensor中的每个元素除以这个数
        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c)
        return pred
        # return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    # 没有用到
    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    # 没有用到，仅在train_rnn.py中使用
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.forward(
                torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State, rHidden_State, rCell_State)
        return pred

    # 没有用到
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            rCell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State, rHidden_State, rCell_State

    # 没有用到
    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            Cell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State
        else:
            Hidden_State = Variable(Hidden_State_data, requires_grad=True)
            Cell_State = Variable(Cell_State_data, requires_grad=True)
            rHidden_State = Variable(Hidden_State_data.cuda(), requires_grad=True)
            rCell_State = Variable(Cell_State_data.cuda(), requires_grad=True)
            return Hidden_State, Cell_State, rHidden_State, rCell_State


class mLSTM(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K  # K=3
        self.A_list = []
        # A是一个邻接矩阵
        # torch.sum(A,0)把A沿第0轴的数全部加起来，得到一个1 * feature_size大小的矩阵
        # torch.diag()则会将一维向量作为对角线构造对角矩阵，其余元素全为0
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        # 矩阵乘法，对邻接矩阵进行度归一化
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A
        # 生成维数为feature_size的单位矩阵
        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                # 让A中的元素最大为1
                A_temp = torch.clamp(A_temp, max=1.)
            # A_list中装入度归一化后的邻接矩阵与单位矩阵相乘的结果
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList(
            # gc_list中包含K个DynamicFilterGNN，区别在于构建时取不同的A_list中的矩阵
            [DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.il = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.kl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.vl = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ql = nn.Linear(gc_input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size, hidden_size)
        # 创建一个feature_size长度的一维张量
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        # 重新设置张量中的值
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        input_size = self.feature_size

        self.rfl = nn.Linear(input_size + hidden_size, hidden_size)
        self.ril = nn.Linear(input_size + hidden_size, hidden_size)
        self.rkl = nn.Linear(input_size + hidden_size, hidden_size)
        self.rvl = nn.Linear(input_size + hidden_size, hidden_size)
        self.rql = nn.Linear(input_size + hidden_size, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, Normal_State=None, Cell_State=None, rNormal_State=None, rCell_State=None):
        # TODO: 需要将生成Cell_State的步骤按照原本方式修改
        batch_size, time_steps, _ = input.shape
        if Normal_State is None:
            Normal_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if rNormal_State is None:
            rNormal_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        if rCell_State is None:
            rCell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        # 将这四个张量增加一维并将序号为1的维度增加为time_steps
        Normal_State = Normal_State.unsqueeze(1).expand(-1, time_steps, -1)
        Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)
        rNormal_State = rNormal_State.unsqueeze(1).expand(-1, time_steps, -1)
        rCell_State = rCell_State.unsqueeze(1).expand(-1, time_steps, -1)
        x = input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            # 沿着第一维拼接gc_list中的模型输出
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Normal_State), 1)
        dim1 = combined.shape[0]
        dim2 = combined.shape[1]
        dim3 = combined.shape[2]
        # 修改combined的维度
        combined = combined.view(dim1, dim2 // 4, dim3 * 4)
        ft = torch.exp(self.fl(combined))  # 线性层也相当于一个权重矩阵
        it = torch.exp(self.il(combined))
        kt = self.kl(combined) / (self.hidden_size ** 0.5)
        vt = self.vl(combined)
        qt = self.ql(combined)
        ot = torch.sigmoid(self.ol(combined))
        Normal_State = Normal_State * ft + it * kt
        Cell_State = Cell_State * ft + kt * vt * it
        ht = Cell_State * qt / max(torch.sum(Normal_State * qt), 1)  # Normal_State(batch_size, step, feature_size)
        Hidden_State = ht * ot
        rcombined = torch.cat((input, rNormal_State), 1)
        d1 = rcombined.shape[0]
        d2 = rcombined.shape[1]
        d3 = rcombined.shape[2]
        # rcombined只经过一次拼接得到，所以这里数字是2
        rcombined = rcombined.view(d1, d2 // 2, d3 * 2)
        rft = torch.exp(self.rfl(rcombined))  # 线性层也相当于一个权重矩阵
        rit = torch.exp(self.ril(rcombined))
        rkt = self.rkl(rcombined) / (self.hidden_size ** 0.5)
        rvt = self.rvl(rcombined)
        rqt = self.rql(rcombined)
        rot = torch.sigmoid(self.rol(rcombined))
        rNormal_State = rNormal_State * rft + rit * rkt
        rCell_State = rCell_State * rft + rkt * rvt * rit
        ht = rCell_State * rqt / max(torch.sum(rNormal_State * rqt), 1)
        rHidden_State = ht * rot
        # Kalman Filtering
        # torch.var实际为计算张量所有元素的样本方差,
        var1, var2 = torch.var(input), torch.var(gc)
        # tensor与一个数进行/运算，实际为tensor中的每个元素除以这个数
        pred = (Hidden_State * var1 * self.c + rHidden_State * var2) / (var1 + var2 * self.c)
        return pred
        # return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred


class sLSTM_Block(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True, num_layer=2):
        super().__init__()
        self.layers = nn.ModuleList([sLSTM(K, A, feature_size, Clamp_A) for _ in range(num_layer)])
        self.gc_list = self.layers[0].gc_list

    def forward(self, input):
        hidden_state, normal_state, cell_state = None, None, None
        for layer in self.layers:
            hidden_state, normal_state, cell_state = layer(input, hidden_state, normal_state, cell_state)
        return hidden_state


class sLSTM(nn.Module):
    def __init__(self, K, A, feature_size, Clamp_A=True):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K  # K=3
        self.A_list = []
        # A是一个邻接矩阵
        # torch.sum(A,0)把A沿第0轴的数全部加起来，得到一个1 * feature_size大小的矩阵
        # torch.diag()则会将一维向量作为对角线构造对角矩阵，其余元素全为0
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        # 矩阵乘法，对邻接矩阵进行度归一化
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A
        # 生成维数为feature_size的单位矩阵
        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                # 让A中的元素最大为1
                A_temp = torch.clamp(A_temp, max=1.)
            # A_list中装入度归一化后的邻接矩阵与单位矩阵相乘的结果
            self.A_list.append(A_temp)

        self.gc_list = nn.ModuleList(
            # gc_list中包含K个DynamicFilterGNN，区别在于构建时取不同的A_list中的矩阵
            [DynamicFilterGNN(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])
        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.wf = nn.Linear(gc_input_size + hidden_size, hidden_size, bias=False)
        self.rf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.wi = nn.Linear(gc_input_size + hidden_size, hidden_size, bias=False)
        self.ri = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.wz = nn.Linear(gc_input_size + hidden_size, hidden_size, bias=False)
        self.rz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bz = nn.Parameter(torch.zeros(hidden_size))
        self.wo = nn.Linear(gc_input_size + hidden_size, hidden_size, bias=False)
        self.ro = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        # 创建一个feature_size长度的一维张量
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        # 重新设置张量中的值
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

    def forward(self, input, Hidden_State=None, Normal_State=None, Cell_State=None):
        # TODO: 首先使用单纯的sLSTM加上本来的计算NC的方法实现前向传播，也就是去掉residual的部分
        batch_size, time_steps, _ = input.shape
        if Hidden_State is None:
            Hidden_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
            # 将这四个张量增加一维并将序号为1的维度增加为time_steps
            Hidden_State = Hidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        if Normal_State is None:
            Normal_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
            Normal_State = Normal_State.unsqueeze(1).expand(-1, time_steps, -1)
        if Cell_State is None:
            Cell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
            Cell_State = Cell_State.unsqueeze(1).expand(-1, time_steps, -1)
        # if rHidden_State is None:
        #     rHidden_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        # if rNormal_State is None:
        #     rNormal_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())
        # if rCell_State is None:
        #     rCell_State = Variable(torch.zeros(batch_size, self.feature_size).cuda())

        # rHidden_State = rHidden_State.unsqueeze(1).expand(-1, time_steps, -1)
        # rNormal_State = rNormal_State.unsqueeze(1).expand(-1, time_steps, -1)
        # rCell_State = rCell_State.unsqueeze(1).expand(-1, time_steps, -1)
        x = input
        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            # 沿着第一维拼接gc_list中的模型输出
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Normal_State), 1)
        dim1 = combined.shape[0]
        dim2 = combined.shape[1]
        dim3 = combined.shape[2]
        # 修改combined的维度
        combined = combined.view(dim1, dim2 // 4, dim3 * 4)
        ft = torch.sigmoid(self.wf(combined) + self.rf(Hidden_State) + self.bf)
        it = torch.exp(self.wi(combined) + self.ri(Hidden_State) + self.bi)
        # φ()暂时认为是tanh激活函数
        zt = torch.tanh(self.wz(combined) + self.rz(Hidden_State) + self.bz)
        ot = torch.sigmoid(self.wo(combined) + self.ro(Hidden_State) + self.bo)
        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).cuda(), self.Neighbor_weight))
        Cell_State = NC * ft + it * zt
        Normal_State = Normal_State * ft + it
        Hidden_State = ot * (Cell_State / Normal_State)
        # TODO:更换block模式记得修改此处
        # return Hidden_State
        return Hidden_State, Normal_State, Cell_State
        # return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred


# Mamba Network
# 实际ModelArgs就是一个类，存储模型的相关数据信息
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    features: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    K: int = 3
    A: torch.Tensor = None
    feature_size: int = None
    kfgn_mode: str = 'lstm'

    # __post_init__在_init_函数后被调用
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)  # 2 * feature_size
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class KFGN_Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        if self.args.kfgn_mode == 'lstm':
            self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size)
            print('lstm mode')
        elif args.kfgn_mode == 'slstm':
            self.kfgn = sLSTM(K=args.K, A=args.A, feature_size=args.feature_size)
            print('slstm mode')
        elif args.kfgn_mode == 'slstm_block':
            self.kfgn = sLSTM_Block(K=args.K, A=args.A, feature_size=args.feature_size)
            print('slstm_block mode')
        else:
            print('mLstm mode')
            self.kfgn = mLSTM(K=args.K, A=args.A, feature_size=args.feature_size)
        self.encode = nn.Linear(args.features, args.d_model)
        self.encoder_layers = nn.ModuleList([ResidualBlock(args, self.kfgn) for _ in range(args.n_layer)])
        self.encoder_norm = RMSNorm(args.d_model)
        # Decoder (identical to Encoder)
        ##self.decoder_layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)]) #You can optionally uncommand these lines to use the identical Decoder.
        ##self.decoder_norm = RMSNorm(args.d_model) #You can optionally uncommand these lines to use the identical Decoder.
        self.decode = nn.Linear(args.d_model, args.features)

    def forward(self, input_ids):
        # input_ids(batch_size, step, features)
        x = self.encode(input_ids)
        # x(batch_size, step, features)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        # Decoder
        ##for layer in self.decoder_layers:#You can optionally uncommand these lines to use the identical Decoder.
        ##    x = layer(x) #You can optionally uncommand these lines to use the identical Decoder.
        ##x = self.decoder_norm(x) #You can optionally uncommand these lines to use the identical Decoder.

        # Output
        x = self.decode(x)

        return x


# Residual Block in Mamba Model
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs, kfgn):
        super().__init__()
        self.args = args
        # self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size)
        if args.kfgn_mode == 'lstm':
            self.kfgn = KFGN(K=args.K, A=args.A, feature_size=args.feature_size)
        elif args.kfgn_mode == 'slstm':
            self.kfgn = sLSTM(K=args.K, A=args.A, feature_size=args.feature_size)
        elif args.kfgn_mode == 'slstm_block':
            self.kfgn = sLSTM_Block(K=args.K, A=args.A, feature_size=args.feature_size)
        else:
            self.kfgn = mLSTM(K=args.K, A=args.A, feature_size=args.feature_size)
        self.mixer = MambaBlock(args, kfgn)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x0 = x
        x1 = self.norm(x)
        x2 = self.kfgn(x1)
        x3 = self.mixer(x2)
        output = x3 + x1

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs, kfgn):
        super().__init__()
        self.args = args
        self.kfgn = kfgn
        # d_inner = feature_size * 2, d_model = feature_size
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        # 一维卷积
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,  # 2 * feature_size
            out_channels=args.d_inner,  # 2 * feature_size
            bias=args.conv_bias,
            kernel_size=args.d_conv,  # 4
            groups=args.d_inner,  # 2 * feature_size,权重张量变为(out_channels // groups, in_channels // groups, kernel_size)
            padding=args.d_conv - 1,
        )
        # d_state=16
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        # 生成[1, d_state]的正整数一维张量，并重复d_inner行得到二维向量A(d_inner,d_state)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))  # 计算自然对数
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        # 投影后的结果拆分为两个矩阵
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        # rearrange进行维度转换时有转置的效果，与reshape不同
        x = rearrange(x, 'b l d_in -> b d_in l')

        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)
        # y shape (b, l, d_in)
        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape  # (d_inner, d_state = 16)

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        # softplus(x) = log(1+exp(x))
        # delta(batch_size, step, d_inner)
        # A(d_inner, d_state)
        # B,C(batch_size, step, d_state)
        # D(d_inner)
        y = self.selective_scan(x, delta, A, B, C, D)
        # y shape (b, l, d_in)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        # d_in = 2 * feature_size
        (b, l, d_in) = u.shape
        # n = d_state = 16
        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        # In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        # temp_adj(feature_size, feature_size)
        temp_adj = self.kfgn.gc_list[-1].get_transformed_adjacency()
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)
        # temp_adj的数据被填充到左上角
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj
        # delta_p(batch_size, step, d_in)
        delta_p = torch.matmul(delta, temp_adj_padded)

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        # deltaA(b, l, d_in, d_state)
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        # deltaB_u(b, l, d_in, d_state)
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            # deltaA与deltaB_u沿第二维切片并与x相乘
            # x(b, d_in, d_state)
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # C(b, l, n)
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D
        # y shape (b, l, d_in)
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        # eps: 一个小的常数，用于防止除以零。默认情况下，它是1e-5。

        return output
