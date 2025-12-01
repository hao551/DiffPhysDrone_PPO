import torch
from torch import nn

def g_decay(x, alpha):
    return x * alpha + x.detach() * (1 - alpha)

# # 在 main_cuda.py 中
# if args.no_odom:
#     model = Model(7, 6)      # 状态7维（[3] 目标速度向量，[3] 无人机的上向轴即机体朝向，[1] 目标方向），动作6维
# else:
#     model = Model(7+3, 6)    # 状态10维（上面的再机上odom），动作6维
class Model(nn.Module):
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        super().__init__()
        #轻量级CNN：3层卷积，逐步减小空间分辨率，增加通道数
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),  # 1, 12, 16 -> 32, 6, 8
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False), #  32, 6, 8 -> 64, 4, 6
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False), #  64, 4, 6 -> 128, 2, 4
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128*2*4, 192, bias=False),#最终压缩：输出192维特征向量
        )
        self.v_proj = nn.Linear(dim_obs, 192)#将低维状态向量映射到与视觉特征相同的维度（192）
        self.v_proj.weight.data.mul_(0.5)#权重初始化：乘以0.5，降低初始权重幅度，稳定训练

        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        pass
    
    # x：深度图像 [B, 1, 12, 16]（经过预处理的单通道深度图）
    # v：低维状态向量 [B, dim_obs]（目标方向、姿态、安全裕度等）
    # hx：GRU隐状态 [B, 192]（可选，用于序列建模）
    def forward(self, x: torch.Tensor, v, hx=None):
        img_feat = self.stem(x)
        x = self.act(img_feat + self.v_proj(v))#元素级相加：视觉特征 + 状态特征
        hx = self.gru(x, hx)
        act = self.fc(self.act(hx))
        return act, None, hx


if __name__ == '__main__':
    Model()
