import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入特征图转化为列矩阵
    
    Parameters:
    - input_data: numpy.array, (B, C, H, W)
    - filter_h: int, 卷积核高度
    - filter_w: int, 卷积核宽度
    - stride: int, 步长
    - pad: int, 填充
    
    Returns:
    - col: numpy.array, (B*H_out*W_out, C*filter_h*filter_w)
    """
    B, C, H, W = input_data.shape
    H_out = (H + 2*pad - filter_h) // stride + 1
    W_out = (W + 2*pad - filter_w) // stride + 1
    
    # 填充
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((B, C, filter_h, filter_w, H_out, W_out))
    
    # 填充列矩阵
    for y in range(filter_h):
        y_max = y + stride * H_out
        for x in range(filter_w):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    # 重塑为二维矩阵
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B*H_out*W_out, -1)
    return col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        r'''
        卷积层的初始化

        Parameter:
        - W: numpy.array, (C_out, C_in, K_h, K_w)
        - b: numpy.array, (C_out)
        - stride: int
        - pad: int
        '''
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        r'''
        卷积层的前向传播

        Parameter:
        - x: numpy.array, (B, C, H, W)

        Return:
        - y: numpy.array, (B, C', H', W')
             H' = (H - Kh + 2P) / S + 1
             W' = (W - Kw + 2P) / S + 1
        '''
        ########## Begin ##########
        # 获取输入数据的维度：批次大小B，通道数C，高度H，宽度W
        B, C, H, W = x.shape
        # 获取卷积核的维度：输出通道数C_out，输入通道数C_in，卷积核高度Kh，卷积核宽度Kw
        C_out, C_in, Kh, Kw = self.W.shape
        # 获取步长S和填充P
        S = self.stride
        P = self.pad
        # 计算输出特征图的高度H_out
        H_out = (H - Kh + 2 * P) // S + 1
        # 计算输出特征图的宽度W_out
        W_out = (W - Kw + 2 * P) // S + 1
        # 使用im2col将输入数据转换为列矩阵，便于矩阵乘法运算
        x_col = im2col(x, Kh, Kw, S, P)
        # 将卷积核权重重塑为二维矩阵，每行对应一个输出通道的所有权重
        W_col = self.W.reshape(C_out, -1)
        # 计算卷积结果：输入列矩阵与权重转置的矩阵乘法，加上偏置
        y_col = np.dot(x_col, W_col.T) + self.b
        # 将结果重塑为(B, H_out, W_out, C_out)并转置为(B, C_out, H_out, W_out)
        y = y_col.reshape(B, H_out, W_out, C_out).transpose(0, 3, 1, 2)
        return y
        ########## End ##########

