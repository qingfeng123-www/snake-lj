import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入特征图转化为列矩阵
    
    Parameters:
    - input_data: numpy.array, (B, C, H, W)
    - filter_h: int, 池化窗口高度
    - filter_w: int, 池化窗口宽度
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


class MaxPool:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        r'''
        池化层的初始化

        Parameter:
        - pool_h: int
        - pool_h: int
        - stride: int
        - pad: int
        '''
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        r'''
        池化层的前向传播

        Parameter:
        - x: numpy.array, (B, C, H, W)

        Return:
        - y: numpy.array, (B, C, H', W')
             H' = (H - Kh + 2P) / S + 1
             W' = (W - Kw + 2P) / S + 1
        '''
        ########## Begin ##########
        B, C, H, W = x.shape
        pool_h = self.pool_h
        pool_w = self.pool_w
        stride = self.stride
        pad = self.pad
        
        # 计算输出特征图尺寸
        H_out = (H - pool_h + 2 * pad) // stride + 1
        W_out = (W - pool_w + 2 * pad) // stride + 1
        
        # 对输入进行im2col操作
        x_col = im2col(x, pool_h, pool_w, stride, pad)
        
        # 对每个通道的池化窗口取最大值
        x_col = x_col.reshape(-1, C, pool_h * pool_w)
        y_col = np.max(x_col, axis=2)
        
        # 将结果重塑为正确的输出形状
        y = y_col.reshape(B, H_out, W_out, C).transpose(0, 3, 1, 2)
        
        return y
        ########## End ##########
