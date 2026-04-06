import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        r'''
        Sigmoid激活函数的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, d1, d2, ..., dk)
        '''
        ########## Begin ##########
        # 计算Sigmoid函数的输出
        self.out = 1 / (1 + np.exp(-x))
        return self.out

        ########## End ##########


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        r'''
        ReLU激活函数的前向传播。

        Parameter:
        - x: numpy.array, (B, d1, d2, ..., dk)

        Return:
        - y: numpy.array, (B, d1, d2, ..., dk)
        '''
        ########## Begin ##########
        # 计算ReLU激活函数的输出
        self.out = np.maximum(0, x)
        return self.out
        ########## End ##########
