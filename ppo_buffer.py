from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Union, Iterable

import torch
from torch import nn

@dataclass
class MiniBatch:
    depth: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor#旧策略（rollout 时的策略）对 action 的 log 概率：old_logprob。
    value: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor
    hx: torch.Tensor

# 定义一个“采样缓存”类，用来：在 rollout 过程中把每一步的数据都存起来；
# rollout 结束后，算优势 / 回报；然后按 mini‑batch / 序列形式把数据喂给 PPO 更新。
class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, hidden_size: int, device: torch.device) -> None:
        self.num_steps = num_steps#每次 rollout 的时间步数 T。
        self.num_envs = num_envs#并行环境数量 B（batch_size）。
        self.hidden_size = hidden_size#GRU 的隐藏状态维度 H。
        self.device = device
        self.clear()

    # 清空缓存
    def clear(self) -> None:
        
        self.depths_list: List[torch.Tensor] = []
        self.states_list: List[torch.Tensor] = []
        self.actions_list: List[torch.Tensor] = []
        self.logprobs_list: List[torch.Tensor] = []
        self.values_list: List[torch.Tensor] = []
        self.rewards_list: List[torch.Tensor] = []
        self.dones_list: List[torch.Tensor] = []
        self.hx_list: List[torch.Tensor] = []
        self.storage_prepared = False

    def add(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        hx: torch.Tensor,
    ) -> None:
        self.depths_list.append(depth.detach())
        self.states_list.append(state.detach())
        self.actions_list.append(action.detach())
        self.logprobs_list.append(logprob.detach())
        self.values_list.append(value.detach())
        self.rewards_list.append(reward.detach())
        self.dones_list.append(done.detach())
        if hx is not None:
            self.hx_list.append(hx.detach())
        else:
            zeros = torch.zeros(depth.shape[0], self.hidden_size, device=self.device)
            self.hx_list.append(zeros)

    def _stack(self) -> None:
        if self.storage_prepared:
            return
        self.depths = torch.stack(self.depths_list)  # [T, B, C, H, W]
        self.states = torch.stack(self.states_list)  # [T, B, D]
        self.actions = torch.stack(self.actions_list)  # [T, B, A]
        self.logprobs = torch.stack(self.logprobs_list)  # [T, B]
        self.values = torch.stack(self.values_list)  # [T, B]
        self.rewards = torch.stack(self.rewards_list)  # [T, B]
        self.dones = torch.stack(self.dones_list)  # [T, B]
        self.hxs = torch.stack(self.hx_list)  # [T, B, H]
        self.storage_prepared = True

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        last_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        self._stack()
        T, B = self.rewards.shape
        advantages = torch.zeros((T, B), device=self.device)
        next_advantage = torch.zeros(B, device=self.device)
        next_value = last_values
        next_non_terminal = 1.0 - last_dones.float()
        for step in reversed(range(T)):
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            next_advantage = delta + gamma * gae_lambda * next_non_terminal * next_advantage
            advantages[step] = next_advantage
            next_value = self.values[step]
            next_non_terminal = 1.0 - self.dones[step].float()
        self.advantages = advantages
        self.returns = self.advantages + self.values

    def get_minibatches(self, num_minibatches: int, seq_len: int) -> Iterator[Dict[str, torch.Tensor]]:
        assert self.storage_prepared, "Buffer not prepared. Call compute_returns_and_advantages first."
        T, B = self.rewards.shape#T, B：从 rewards 里取时间步数和 batch 大小。T其实就是num_steps，B就是batch_size。
        
        if T % seq_len != 0:
            raise ValueError(f"num_steps ({T}) must be divisible by seq_len ({seq_len})")
            
        num_seqs_per_env = T // seq_len#每个 env（每架无人机）上能切出多少段序列。
        total_seqs = B * num_seqs_per_env#总共有多少段序列。
        
        if total_seqs % num_minibatches != 0:
            raise ValueError(f"Total sequences ({total_seqs}) must be divisible by num_minibatches")
            
        seqs_per_minibatch = total_seqs // num_minibatches
        indices = torch.randperm(total_seqs, device=self.device)

        # Helper to reshape: [T, B, ...] -> [total_seqs, seq_len, ...]
        def reshape_seq(tensor):
            # [T, B, ...] -> [num_seqs, seq_len, B, ...]
            sh = tensor.shape
            #以 self.depths: [T, B, C, H, W] 为例：reshape(num_seqs_per_env, seq_len, B, C, H, W)
            #把时间维 T 拆成：
            #第 0 维：num_seqs_per_env（这条轨迹上有多少序列）
            #第 1 维：seq_len（每条序列的长度）
            tensor = tensor.reshape(num_seqs_per_env, seq_len, B, *sh[2:])
            # -> [B, num_seqs, seq_len, ...]
            # 变为 [B, num_seqs_per_env, seq_len, C, H, W]
            # 这样第 0 维是“哪一个 env”，第 1 维是“这个 env 上的第几段序列”。
            tensor = tensor.permute(2, 0, 1, *range(3, len(sh) + 1))
            # -> [B * num_seqs, seq_len, ...]
            # 把 B 和 num_seqs_per_env 两个维度拍平成 total_seqs：
            # 每一条序列就是一行：[seq_len, C, H, W]
            # 一共有 total_seqs 行。
            #最终的形状 [6, 2, 3, 4, 5] 表示我们有 6 条序列，每条序列的长度为 2，包含图像的通道、高度和宽度。
            return tensor.reshape(total_seqs, seq_len, *sh[2:])

        # Reshape all data into sequences
        seq_depth = reshape_seq(self.depths)
        seq_state = reshape_seq(self.states)
        seq_action = reshape_seq(self.actions)
        seq_logprob = reshape_seq(self.logprobs)
        seq_value = reshape_seq(self.values)
        seq_adv = reshape_seq(self.advantages)
        seq_returns = reshape_seq(self.returns)
        
        # For hx, we only need the first hidden state of each sequence
        # self.hxs: [T, B, H]
        # Select indices 0, seq_len, 2*seq_len, ...
        hx_indices = torch.arange(0, T, seq_len, device=self.device)#选出每段序列的“起点 step”；
        # [num_seqs, B, H] -> [B, num_seqs, H] -> [total_seqs, H]
        seq_hx = self.hxs[hx_indices].permute(1, 0, 2).reshape(total_seqs, -1)#选出每段序列的“起点 step”；

        for start in range(0, total_seqs, seqs_per_minibatch):
            mb_idx = indices[start : start + seqs_per_minibatch]
            yield {
                "depth": seq_depth[mb_idx],     # [MB, seq_len, ...]
                "state": seq_state[mb_idx],     # [MB, seq_len, ...]
                "action": seq_action[mb_idx],   # [MB, seq_len, ...]
                "logprob": seq_logprob[mb_idx], # [MB, seq_len]
                "value": seq_value[mb_idx],     # [MB, seq_len]
                "advantage": seq_adv[mb_idx],   # [MB, seq_len]
                "returns": seq_returns[mb_idx], # [MB, seq_len]
                #seq_hx 被切成 mini‑batch，作为 initial_hx 返回。
                "initial_hx": seq_hx[mb_idx],   # [MB, hidden_size]
            }

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        #处理 input_shape 的输入格式，确保它最终统一成 torch.Size 对象
        self.input_shape = (
            #如果传进来的是一个可迭代对象（比如 list/tuple/torch.Size），就说明本来就是一个“形状”，直接拿来转成 torch.Size
            #如果传进来的是一个 整数，说明它只是一个维度大小（比如 1），需要把它包成一个 tuple 才能变成形状。
            #可迭代对象 = 能用 for ... in ... 遍历的对象。判断标准 = 实现了 __iter__() 或 __getitem__() 方法。
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            #例如你传的是 1，那它实际想表达的是「最后一维是 1」，所以包装成 torch.Size((1,))。
            #若直接传整数会报错。因此遇到整数时先包成元组 ((input_shape,),) 再构造。
            else torch.Size((input_shape,))
        )
        #beta：指数滑动平均的衰减因子（越接近 1 越平滑，越慢更新）。
        #epsilon：防止除零 / 太小方差导致数值爆炸。
        self.epsilon = epsilon
        self.beta = beta

        #running_mean：滑动平均的均值估计 E[x]
        #running_mean_sq：滑动平均的平方均值估计 E[x^2]
       #debiasing_term：EMA 的「去偏因子」，用来修正刚开始时 EMA 偏向 0 的问题。
        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        #register_buffer的好处是：这些变量会随着模型一起保存/加载（state_dict），也会随着模型.to(device)移动设备，但它们不是模型的可训练参数。
        #不会进入反向传播计算图，也不会被优化器更新。
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    #需要当前的均值、方差时才调用
    def running_mean_var(self):
        #self.running_mean 其实存的是：mt​≈(1−β)k=1∑t​βt−kxk​
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        #去偏平方根号的均值
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        #方差估计
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    #用新的一批数据更新均值和方差的滑动统计量，保证随着训练进程，全局的均值/方差估计能够平滑跟踪数据分布的非平稳变化
    #ValueNorm 用指数滑动平均在线估计均值/方差，供 normalize/denormalize 使用，常用于把回报/价值标准化，提升训练稳定性。
    def update(self, input_vector: torch.Tensor):
        #取输入张量 最后和 self.input_shape 等长的维度。
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        #input_vector.dim() - len(self.input_shape) = batch 维度的个数，range为代表batch的维度，接下来在这些维度上求均值
        #input_vector.dim()返回张量的维度个数（多少个轴）。例如 torch.randn(32, 10, 3) → dim() = 3
        
        #input_vector.dim() - len(self.input_shape) = batch 维度的个数，range为代表batch的维度，接下来在这些维度上求均值
        #比如该项目里用的T,B维度求的均值
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        #更新数据
        #running_mean=β⋅running_mean+(1−β)⋅batch_mean
        #目的：用指数滑动平均(EMA)在线估计“回报/价值”的均值与方差，做到低内存(O(1))、抗噪声、适应非平稳分布，比“用当前小批次统计”稳定很多。
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        #debiasing_term←β⋅debiasing_term+(1−β)⋅1
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))
