import torch
from torch import nn
from torch.distributions import Independent, Beta
import torch.nn.functional as F


class PPONetwork(nn.Module):
    """
    Actor-critic network that reuses the lightweight CNN + GRU backbone
    from the supervised model but exposes separate actor (Gaussian)
    and critic heads for PPO.
    """

    def __init__(
        self,
        dim_obs: int,
        dim_action: int,
        hidden_size: int = 192,
        img_channels: int = 1,
    ) -> None:
        super().__init__()
        self.dim_action = dim_action
        self.hidden_size = hidden_size

        # CNN stem copied from model.py
        #作用：把深度图像提取成一个长度为 hidden_size 的视觉特征向量。
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, hidden_size, bias=False),
        )
        #把低维状态向量（如目标速度、姿态、margin 等）投影到同样的 hidden_size 维度。
        self.v_proj = nn.Linear(dim_obs, hidden_size)
        #手动把权重乘 0.5，相当于减小初始权重的幅度。
        self.v_proj.weight.data.mul_(0.5)

        #接收融合后的特征（192 维），维护一个同样 192 维的隐状态。用于建模时间上的依赖（历史观测）。
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        #LeakyReLU(0.05)：激活函数，用于引入非线性。
        self.activation = nn.LeakyReLU(0.05)

        #性能主要由策略的形状决定。这里输出 Beta 分布的 alpha 和 beta 两组参数。
        #因此需要输出 2 * dim_action 维度。
        self.actor = nn.Linear(hidden_size, dim_action * 2, bias=False)
        #手动把权重乘 0.01，相当于减小初始权重的幅度。
        self.actor.weight.data.mul_(0.01)
        #输出状态价值 V(s)。用于 critic，参与 PPO 的 value loss。
        self.value_head = nn.Linear(hidden_size, 1)

    @torch.no_grad()
    def reset(self) -> None:
        """Reset method kept for API parity."""
        pass

    def _forward_backbone(self, depth: torch.Tensor, state: torch.Tensor, hx: torch.Tensor | None):
        feat = self.stem(depth)
        feat = self.activation(feat + self.v_proj(state))
        if hx is None:#hx：GRU 的隐藏状态。如果是 None 就在这里初始化。
            hx = depth.new_zeros(depth.shape[0], self.hidden_size)
        #hx_new = gru(x_t, hx_old)
        #nn.GRUCell(input_size, hidden_size) 的文档里写得很清楚（翻译一下）：
        # x 的形状：(batch_size, input_size)
        # h 的形状：(batch_size, hidden_size)
        # 输出 h_new：形状同 h，也是 (batch_size, hidden_size)
        # 只要求 feat 的 最后一维 是 192
        # 只要求 hx 的 最后一维 是 192
        # 前面的那个维度（batch）PyTorch 自动按“多个样本并行”处理
        #其实其他的网络定义也是如此
        hx = self.gru(feat, hx)
        return hx

    def forward(self, depth: torch.Tensor, state: torch.Tensor, hx: torch.Tensor | None = None):
        #输入：深度图 depth、低维状态 state 和上一步的隐藏状态 hx
        hx = self._forward_backbone(depth, state, hx)
        feat = self.activation(hx)

        # 输出 Beta 分布的参数：alpha 和 beta，形状 [B, 2 * dim_action]
        raw = self.actor(feat)
        # 使用 softplus 保证 alpha、beta 为正数，并整体平移到 > 1，避免一开始就是 U 型分布
        raw = F.softplus(raw) + 1.0
        alpha, beta = raw.chunk(2, dim=-1)

        #输出形状先是 [B, 1]，squeeze(-1) 后变成 [B] 表示每个状态的标量价值函数 V(s)
        value = self.value_head(feat).squeeze(-1)
        return (alpha, beta), value, hx

    def dist(self, params):
        alpha, beta = params
        base = Beta(alpha, beta)
        return Independent(base, 1)

    def get_dist(self, depth: torch.Tensor, state: torch.Tensor, hx: torch.Tensor | None = None):
        params, value, next_hx = self.forward(depth, state, hx)
        return self.dist(params), value, next_hx

    def act(self, depth: torch.Tensor, state: torch.Tensor, hx: torch.Tensor | None = None):
        dist, value, next_hx = self.get_dist(depth, state, hx)
        # Beta 分布采样的动作范围在 [0, 1]
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, next_hx, dist.entropy()

