import argparse
import math
import os
import random
from collections import defaultdict, deque
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from env_cuda import Env
from ppo_buffer import RolloutBuffer, ValueNorm
from ppo_model import PPONetwork


def parse_args():
    parser = argparse.ArgumentParser(description="PPO trainer for DiffPhysDrone")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_updates", type=int, default=4000)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--clip_range", type=float, default=0.15)
    parser.add_argument("--clip_vloss", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--linear_lr", action="store_true", default=True)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length for RNN training")
    parser.add_argument("--log_dir", type=str, default="runs/ppo")
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--ctl_dt_mean", type=float, default=1 / 15)
    parser.add_argument("--ctl_dt_std", type=float, default=0.1 / 15)
    parser.add_argument("--act_lag", type=int, default=1)
    parser.add_argument("--disable_partial_reset", action="store_true")
    # 稀疏成功奖励：episode 成功结束时一次性加的 bonus
    parser.add_argument("--success_reward", type=float, default=20.0)
    parser.add_argument("--success_radius", type=float, default=1)
    parser.add_argument("--collision_penalty", type=float, default=2.0)
    parser.add_argument("--progress_coef", type=float, default=3.0)
    parser.add_argument("--progress_clip", type=float, default=2.0)
    parser.add_argument("--avoid_velocity_scale", type=float, default=135.0)
    parser.add_argument("--jerk_scale", type=float, default=15.0)
    parser.add_argument("--snap_scale", type=float, default=15.0 ** 2)

    # reward weights mirroring the supervised losses
    parser.add_argument("--coef_v", type=float, default=4.0)
    parser.add_argument("--coef_speed", type=float, default=0.0)
    parser.add_argument("--coef_v_pred", type=float, default=2.0)
    parser.add_argument("--coef_collide", type=float, default=0.3)
    parser.add_argument("--coef_obj_avoidance", type=float, default=0.1)
    parser.add_argument("--coef_d_acc", type=float, default=0.02)
    parser.add_argument("--coef_d_jerk", type=float, default=0.0005)
    parser.add_argument("--coef_d_snap", type=float, default=0.0)
    parser.add_argument("--coef_ground_affinity", type=float, default=0.0)
    parser.add_argument("--coef_bias", type=float, default=0.01)
    parser.add_argument("--coef_alive", type=float, default=0.05, help="Bonus for staying alive per step")
    parser.add_argument("--coef_goal_bonus", type=float, default=2.0, help="Bonus for getting closer to the goal")

    # environment flags
    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--speed_mtp", type=float, default=1.0)
    parser.add_argument("--fov_x_half_tan", type=float, default=0.53)
    parser.add_argument("--cam_angle", type=int, default=10)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--ground_voxels", action="store_true")
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--random_rotation", action="store_true")
    parser.add_argument("--yaw_drift", action="store_true")
    #只有你在命令行加 --no_odom 时，args.no_odom 才会变成 True
    parser.add_argument("--no_odom", action="store_true")
    
    return parser.parse_args()

#这里是在把环境里的姿态矩阵 env.R，加工成一个“只考虑水平朝向（yaw）的机体系旋转矩阵”，
#方便后面把速度、目标速度投到这个坐标系里当观测。
#把原来的姿态矩阵 env.R，投影成「只包含 yaw、没有 pitch/roll 的旋转矩阵」，
#也就是一个“平面化后的机体朝向矩阵”。
def build_rotation(env: Env) -> torch.Tensor:
    fwd = env.R[:, :, 0].clone()#机体前向轴（机体系 X 轴）在世界坐标中的方向向量。把 pitch/roll 引起的“抬头/压头、翻滚”都去掉了。
    up = torch.zeros_like(fwd)
    #把“前向轴”投到水平面，只保留 yaw
    fwd[:, 2] = 0
    #up 被设成世界坐标的“全局向上” [0, 0, 1]，
    #不随机体 pitch/roll 倾斜，保证 z 轴就是“竖直向上”。
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    left = torch.cross(up, fwd, dim=-1)
    return torch.stack([fwd, left, up], -1)


#本质上是在构造 RL 的观测向量，以及一些后面算 loss / reward 会用到的量。
def build_observation(env: Env, ctl_dt: float, args, yaw_drift: Optional[torch.Tensor]):
    depth, _ = env.render(ctl_dt)
    depth = 3 / depth.clamp(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
    depth = F.max_pool2d(depth[:, None], 4, 4)

    R_body = build_rotation(env)

    target_v_raw = env.p_target - env.p.detach()
    if args.yaw_drift and yaw_drift is not None:
        target_v_raw = torch.squeeze(target_v_raw[:, None] @ yaw_drift, 1)
    target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
    target_v_norm = torch.clamp(target_v_norm, min=1e-3)
    target_v_unit = target_v_raw / target_v_norm
    target_speed = torch.minimum(target_v_norm, env.max_speed)
    target_v_world = target_v_unit * target_speed
    target_v_body = torch.squeeze(target_v_world[:, None] @ R_body, 1)

    local_v = torch.squeeze(env.v[:, None] @ R_body, 1)
    #nv.R[:, 2] 会返回 每个旋转矩阵的第三列（即列索引为 2 的列）
    state_parts = [target_v_body, env.R[:, 2], env.margin[:, None]]
    if not args.no_odom:
        state_parts.insert(0, local_v)
    state_vec = torch.cat(state_parts, -1)

    goal_distance = torch.norm(env.p_target - env.p, 2, -1)
    return {
        "depth": depth,
        "state": state_vec,
        "R_body": R_body,
        "target_v_world": target_v_world.detach(),
        "target_dir_world": target_v_unit.detach(),
        "target_speed": target_speed.squeeze(-1).detach(),
        "target_v_raw": target_v_raw.detach(),
        "goal_distance": goal_distance.detach(),
        "target_v_unit": target_v_unit.detach(),
    }

#把网络输出的动作（机体系坐标）变换成物理环境需要的推力指令（世界坐标）。
def convert_action(action: torch.Tensor, R_body: torch.Tensor, env: Env):
    B = action.shape[0]
    #第 1 维（3）：前/左/上三个分量
    #第 2 维（2）：第 0 列是加速度，第 1 列是速度预测
    action_matrix = action.reshape(B, 3, -1)
    #由于变量都是按列存，所以左乘 R_body
    world_matrix = torch.matmul(R_body, action_matrix)
    a_pred, v_pred = world_matrix.unbind(-1)
    act_cmd = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
    return act_cmd, a_pred, v_pred


ATTRS_TO_COPY: Iterable[str] = [
    "p",
    "v",
    "a",
    "R",
    "R_old",
    "p_old",
    "p_target",
    "act",
    "dg",
    "v_wind",
    "balls",
    "voxels",
    "cyl",
    "cyl_h",
    "margin",
    "max_speed",
    "pitch_ctl_delay",
    "yaw_ctl_delay",
    "thr_est_error",
    "drag_2",
    "z_drag_coef",
]

#这个函数不需要梯度，因为它只是在修改环境状态，不参与反向传播。只重置 done 的无人机
#template_env：一个"模板环境"，用来生成新的随机状态。
#mask：形状 [B]，True 表示这架无人机 done 了（碰撞或成功），需要重置。
#act_buffer：控制指令的延迟缓冲区（deque），也需要同步重置。
@torch.no_grad()
def partial_reset(env: Env, template_env: Env, mask: torch.Tensor, act_buffer):
    if mask is None or mask.sum() == 0:
        return
    #.squeeze(-1)：变成 [num_done]，就是需要重置的 batch index。如果没有任何 done，直接返回。
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return

    template_env.reset()#随机生成新的障碍物、起终点等
    for attr in ATTRS_TO_COPY:
        if not hasattr(env, attr):
            continue
        #getattr(env, attr) 相当于 env.p / env.v / env.balls .
        data = getattr(env, attr) # env 当前的属性，比如 env.p
        tmpl = getattr(template_env, attr)# template_env 新生成的，比如 template_env.p
        data[idx] = tmpl[idx]# 只把 done 的那些无人机替换掉
    if act_buffer is not None:
        for buf in act_buffer:
            buf[idx] = env.act[idx].detach()


def prepare_env(args, device):
    env_kwargs = dict(
        fov_x_half_tan=args.fov_x_half_tan,
        single=args.single,
        gate=args.gate,
        ground_voxels=args.ground_voxels,
        scaffold=args.scaffold,
        speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation,
        cam_angle=args.cam_angle,
    )
    env = Env(args.batch_size, 64, 48, args.grad_decay, device, **env_kwargs)
    template_env = Env(args.batch_size, 64, 48, args.grad_decay, device, **env_kwargs)
    return env, template_env


def sample_ctl_dt(args):
    ctl_dt = random.gauss(args.ctl_dt_mean, args.ctl_dt_std)
    return max(1e-3, ctl_dt)


def main():
    args = parse_args()
    args.single = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    env, template_env = prepare_env(args, device)
    obs_dim = 7 if args.no_odom else 10
    model = PPONetwork(obs_dim, 6, hidden_size=args.hidden_size).to(device)
    if args.resume:
        state_dict = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
    optim = AdamW(model.parameters(), lr=args.lr)

    #创建 PPO 的经验回放缓冲区
    #用来存储每个 update 里采集的所有 transitions：
    #args.num_steps（比如 128）：每个 rollout 采集多少步
    #args.batch_size（比如 512）：并行多少架无人机
    #args.hidden_size（比如 192）：RNN 隐状态的维度
    buffer = RolloutBuffer(args.num_steps, args.batch_size, args.hidden_size, device)

    # ==== 新增：价值归一化器（对 returns 做滑动标准化） ====
    value_norm = ValueNorm(1).to(device)

    #用来记录最近 100 个 episode 的总回报，deque(maxlen=100)：双端队列，自动保持最多 100 个元素
    #当插入第 101 个元素时，自动丢弃最老的那个
    recent_returns = deque(maxlen=100)
    
    #用来记录最近 100 个 episode 的长度（持续了多少步）
    #同样是双端队列，自动保持最多 100 个元素
    #用途：计算 episode_length_mean：最近 100 个 episode 的平均长度
    #用于 TensorBoard 日志，监控无人机能飞多久
    recent_lengths = deque(maxlen=100)
    global_step = 0

    #args.num_updates：总共要训练多少轮（比如 10000），采样阶段：收集 num_steps 步的数据
    for update in trange(args.num_updates, desc="PPO", ncols=100):
        # 线性学习率衰减
        if args.linear_lr:
            frac = 1.0 - update / max(1, args.num_updates)
            optim.param_groups[0]["lr"] = args.lr * frac

        # 熵系数调度：前 2/3 训练保持不变，最后 1/3 线性衰减到 0
        # 用 (update + 1) / num_updates 这样最后一轮刚好衰减到 0
        ent_progress = (update + 1) / max(1, args.num_updates)
        if ent_progress <= 2.0 / 3.0:
            ent_coef_now = args.ent_coef
        else:
            decay_frac = (ent_progress - 2.0 / 3.0) / (1.0 / 3.0)
            ent_coef_now = args.ent_coef * max(0.0, 1.0 - decay_frac)

        env.reset()
        model.reset()
        #初始化 RNN 隐状态
        hidden_state = None
        #记录每个 env（无人机）是否 done
        last_done = torch.zeros(args.batch_size, device=device)
        #动作延迟缓冲区
        act_buffer = deque(maxlen=args.act_lag + 1)
        for _ in range(args.act_lag + 1):
            act_buffer.append(env.act.clone().detach())
        prev_action = env.act.clone().detach()
        prev_prev_action = prev_action.clone()
        prev_distance = torch.zeros(args.batch_size, device=device)
        distance_initialized = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
        disable_partial_reset = args.disable_partial_reset

        yaw_drift = None
        if args.yaw_drift:
            drift_av = torch.randn(args.batch_size, device=device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            yaw_drift = torch.stack(
                [
                    torch.cos(drift_av),
                    -torch.sin(drift_av),
                    zeros,
                    torch.sin(drift_av),
                    torch.cos(drift_av),
                    zeros,
                    zeros,
                    zeros,
                    ones,
                ],
                -1,
            ).reshape(args.batch_size, 3, 3)

        ctl_dt = sample_ctl_dt(args)
        obs = build_observation(env, ctl_dt, args, yaw_drift)

        reward_trackers = defaultdict(float)
        collision_count = 0.0
        success_count = 0.0
        step_count = 0
        episode_returns = torch.zeros(args.batch_size, device=device)
        episode_lengths = torch.zeros(args.batch_size, device=device)

        # 统计本次 rollout 里，各种终止方式的 episode 数（按 env 记）
        episodes_success = 0
        episodes_collision = 0
        episodes_other = 0

        for step in range(args.num_steps):
            depth = obs["depth"]
            state_vec = obs["state"]
            hx_in = hidden_state.detach() if hidden_state is not None else None
            #dist：动作分布（正态分布）
            #value：状态价值 [B]
            #next_hx：下一步的 RNN 隐状态
            dist, value, next_hx = model.get_dist(depth, state_vec, hidden_state)
            action = dist.rsample()
            #logprob：动作的对数概率 [B]，用于 PPO loss
            logprob = dist.log_prob(action)
            # a_pred, v_pred = world_matrix.unbind(-1)
            # act_cmd = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
            centered_action = 2.0 * action - 1.0 
            act_cmd, a_pred, v_pred = convert_action(centered_action, obs["R_body"], env)#把机体系动作变换到世界系

            executed_act = act_buffer[0]
            env.run(executed_act, ctl_dt, obs["target_v_raw"])
            act_buffer.append(act_cmd.detach())

            vec_to_pt = env.find_vec_to_nearest_pt()#到最近障碍物的向量 [10, B, 3]（10 个采样点）
            #到障碍物的距离 [10, B]，减去 env.margin（安全边界），负值表示碰撞
            distance = torch.norm(vec_to_pt, 2, -1) - env.margin
            approach = torch.zeros_like(distance)
            if distance_initialized.any():
                approach = (prev_distance - distance) * args.avoid_velocity_scale
                approach = torch.where(distance_initialized, approach, torch.zeros_like(distance))
            v_to_pt = torch.clamp(approach, min=1.0)

            # reduction="none"：不求和，保持 [B, 3]
            # .mean(-1)：对 xyz 三个维度求平均，得到 [B]
            vel_tracking = F.smooth_l1_loss(env.v, obs["target_v_world"], reduction="none").mean(-1)
            target_dir = obs["target_dir_world"]
            #fwd_v：速度在目标方向上的投影
            #逐元素相乘，结果形状仍然是 [B, 3]，再在最后一维求和，变成 [B]，就是每个无人机速度在目标方向上的标量投影 
            fwd_v = torch.sum(env.v * target_dir, -1)
            #bias_loss 惩罚的是「速度在目标方向以外的那一部分」，也就是“横着飞、乱飘”的速度分量；
            #最小化它，就在鼓励无人机的速度尽量贴着通往目标的直线方向。
            #fwd_v[..., None] * target_dir恰好是把 v投影到方向 d^上得到的投影向量，可以这么理解：
            #
            bias_loss = (
                F.mse_loss(env.v, fwd_v[..., None] * target_dir, reduction="none").mean(-1) * 3.0
            )
            #速度大小 loss
            speed_loss = F.smooth_l1_loss(fwd_v, obs["target_speed"], reduction="none")
            #速度预测 loss
            #这个 loss 只用于记录到 TensorBoard，观察趋势不会影响网络的参数更新，因此detach()
            v_pred_loss = F.mse_loss(v_pred.detach(), env.v.detach(), reduction="none").mean(-1)

            # Reduce along sub-division dimension (10 samples), keep per-env shape [B]
            loss_obj_avoidance = (v_to_pt * (1 - distance).relu().pow(2)).mean(0)
            loss_collide = (F.softplus(distance.mul(-32.0)) * v_to_pt).mean(0)
            loss_ground = env.p[:, 2].relu().pow(2)
            loss_d_acc = executed_act.pow(2).sum(-1)
            jerk = (executed_act - prev_action) * args.jerk_scale
            loss_d_jerk = jerk.pow(2).sum(-1)
            #snap = (executed_act - 2 * prev_action + prev_prev_action) * args.snap_scale
            #loss_d_snap = snap.pow(2).sum(-1)

            new_goal_distance = torch.norm(env.p_target - env.p, 2, -1)

            # 原始进度：d_{t-1} - d_t
            raw_progress = obs["goal_distance"] - new_goal_distance
            # 正向进度（靠近目标）
            progress_pos = torch.clamp(raw_progress, 0.0, args.progress_clip)
            # 负向进度（远离目标，取绝对值）
            progress_neg = torch.clamp(-raw_progress, 0.0, args.progress_clip)
            # 日志记录用的净进度
            progress = progress_pos - progress_neg

            # Collision if any sub-division sample is inside 
            #这个是计算每一步的碰撞率，应该要降低到e-4
            collided = (distance < 0).any(0)
            success = (~collided) & (new_goal_distance < args.success_radius)
            #done在这里填入相关状态了
            done = collided | success

            # ========= 极简奖励：只看进度 + 成功/碰撞 =========         
            # 平滑惩罚：只惩罚动作过大 / 抖动
            smooth_penalty = (
                args.coef_d_acc * loss_d_acc
                + args.coef_d_jerk * loss_d_jerk
            )

            collision_shaping = (
                args.coef_obj_avoidance * loss_obj_avoidance  # 或用 args.coef_obj_avoidance，但把默认改小
                + args.coef_collide * loss_collide  # 这个看一下后面要不要加回去
            )

            # 基础进度奖励：朝目标方向前进的距离（已做裁剪）
            # 前进用原系数，后退惩罚系数更小，避免绕障碍物时被过度惩罚
            reward = (
                args.progress_coef * progress_pos
                - 0.3 * args.progress_coef * progress_neg
            )
            # 方向惩罚：速度偏离目标方向越多，reward 越低
            reward -= args.coef_bias * bias_loss
            # 连续时间惩罚：按真实物理时间扣，每经过 ctl_dt 秒就扣 coef_alive * ctl_dt
            # 这样总惩罚大致与 episode 实际飞行时间成正比，比按“步数”更连续
            # time_cost = ctl_dt * args.coef_alive
            # reward -= time_cost
            # 控制平滑惩罚：动作过大 / 抖动
            reward -= smooth_penalty
            # 避障 shaping 惩罚
            reward -= collision_shaping
            # 终止碰撞惩罚：发生碰撞的 episode 在终止那一步一次性扣除
            reward -= args.collision_penalty * collided.float()
            # 稀疏成功奖励：成功到达目标的 episode 在终止那一步一次性加分
            reward += args.success_reward * success.float()

            # 目标附近的连续奖励：距离越近越大，远处近似为 0（纯密集奖励）
            # 0.5 和 3.0 是经验初始值，可根据场景大小和 reward 量级再调
            goal_bonus = args.coef_goal_bonus * torch.exp(-new_goal_distance / 3.0)
            reward += goal_bonus

            buffer.add(depth, state_vec, action, logprob, value, reward, done.float(), hx_in)
            episode_returns += reward
            episode_lengths += 1

            reward_trackers["reward"] += reward.mean().item()
            reward_trackers["loss_v"] += vel_tracking.mean().item()
            reward_trackers["loss_obj"] += loss_obj_avoidance.mean().item()
            reward_trackers["loss_collide"] += loss_collide.mean().item()
            reward_trackers["loss_d_acc"] += loss_d_acc.mean().item()
            reward_trackers["loss_d_jerk"] += loss_d_jerk.mean().item()
            #reward_trackers["loss_d_snap"] += loss_d_snap.mean().item()
            #reward_trackers["loss_alive"] += time_cost
            reward_trackers["loss_bias"] += bias_loss.mean().item()
            reward_trackers["loss_speed"] += speed_loss.mean().item()
            reward_trackers["loss_v_pred"] += v_pred_loss.mean().item()
            reward_trackers["progress"] += progress.mean().item()
            reward_trackers["goal_bonus"] += goal_bonus.mean().item()
            #reward_trackers["distance"] += distance.mean().item()
            collision_count += collided.float().sum().item()
            success_count += success.float().sum().item()
            step_count += 1

            completed = done.nonzero(as_tuple=False).squeeze(-1)
            if completed.numel() > 0:
                # 这些 env 的 episode 在当前 step 结束，统计终止原因
                ep_success = success[completed].float().sum().item()
                ep_collision = collided[completed].float().sum().item()
                episodes_success += ep_success
                episodes_collision += ep_collision
                # 预留“其他终止原因”（比如将来如果有时间截断等）
                episodes_other += completed.numel() - ep_success - ep_collision

                recent_returns.extend(episode_returns[completed].detach().cpu().tolist())
                recent_lengths.extend(episode_lengths[completed].detach().cpu().tolist())
                episode_returns[completed] = 0
                episode_lengths[completed] = 0

            prev_prev_action = prev_action.clone()
            prev_action = executed_act.detach()
            prev_distance = distance.detach()
            distance_initialized[:] = True

            if not disable_partial_reset:
                partial_reset(env, template_env, done, act_buffer)
                if done.any():
                    distance_initialized[done] = False
                    if prev_distance.ndim == 2:
                        prev_distance[:, done] = 0
                    else:
                        prev_distance[done] = 0
                    prev_action[done] = env.act[done].detach()
                    prev_prev_action[done] = env.act[done].detach()
            if next_hx is not None:
                hidden_state = next_hx.detach()
                #相当于下一个时间步对这些环境来说 从“全零隐状态”重新开始，避免上一条轨迹的历史信息泄漏到新 episode 里。
                hidden_state[done] = 0
            else:
                hidden_state = None
            last_done = done.float()

            ctl_dt = sample_ctl_dt(args)
            obs = build_observation(env, ctl_dt, args, yaw_drift)

        with torch.no_grad():
            dist, last_value, _ = model.get_dist(obs["depth"], obs["state"], hidden_state)

        # last_value 是这次 rollout 末尾每个 env 最后状态的critic价值估计，用来在 GAE 里给“没真正结束、只是被时间截断的轨迹”做 bootstrapping。
        buffer.compute_returns_and_advantages(last_value, last_done, args.gamma, args.gae_lambda)

        # 归一化优势（和原来一样）
        adv = buffer.advantages
        buffer.advantages = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # ==== 新增：用 ValueNorm 对 returns 做滑动标准化 ==== 
        # 每个 PPO update 更新一次全局统计量，每次用的是整批 T×B 的数据”，而不是“每个时间步单独 update 一次”。
        # buffer.returns: [T, B]，先在最后一维加一个 size=1 的维度，匹配 ValueNorm(input_shape=1)
        returns_vec = buffer.returns.unsqueeze(-1)          # [T, B, 1]
        #保证在T，B维度上更新均值和方差，注意是两个一块
        value_norm.update(returns_vec)                      # 更新滑动均值 / 方差
        mean, var = value_norm.running_mean_var()           # 得到去偏均值/方差
        # 把标准化后的 returns 写回 buffer，保持形状仍然是 [T, B]
        buffer.returns = ((returns_vec - mean) / torch.sqrt(var)).squeeze(-1)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        approx_kl_total = 0.0
        clip_fraction = 0.0
        nbatches = 0

        for _ in range(args.ppo_epochs):
            for batch in buffer.get_minibatches(args.num_minibatches, args.seq_len):
                # Initial hidden state for the sequence
                hx = batch["initial_hx"]
                
                # Lists to store outputs for each step in the sequence
                new_logprobs = []
                entropies = []
                value_preds = []
                
                # Iterate over the sequence length
                for t in range(args.seq_len):
                    # batch["depth"]: [MB, seq_len, ...] -> select t -> [MB, ...]
                    obs_depth = batch["depth"][:, t]
                    obs_state = batch["state"][:, t]
                    action = batch["action"][:, t]
                    
                    dist, value_pred, hx = model.get_dist(obs_depth, obs_state, hx)
                    
                    new_logprob = dist.log_prob(action)
                    entropy = dist.entropy()
                    
                    new_logprobs.append(new_logprob)
                    entropies.append(entropy)
                    value_preds.append(value_pred)
                
                # Stack and Flatten: [seq_len, MB, ...] -> [MB, seq_len, ...] -> [MB * seq_len, ...]
                # Note: buffer stores as [MB, seq_len], so we should permute to match
                
                new_logprob = torch.stack(new_logprobs, dim=1).flatten(0, 1)
                entropy = torch.stack(entropies, dim=1).flatten(0, 1)
                value_pred = torch.stack(value_preds, dim=1).flatten(0, 1)
                
                # Flatten batch targets to match [MB * seq_len]
                b_logprob = batch["logprob"].flatten(0, 1)
                b_advantage = batch["advantage"].flatten(0, 1)
                b_value = batch["value"].flatten(0, 1)
                b_returns = batch["returns"].flatten(0, 1)

                ratio = (new_logprob - b_logprob).exp()
                surr1 = ratio * b_advantage
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * b_advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                if args.clip_vloss > 0:
                    value_pred_clipped = b_value + (value_pred - b_value).clamp(
                        -args.clip_vloss, args.clip_vloss
                    )
                    value_loss = 0.5 * torch.max(
                        (value_pred - b_returns).pow(2),
                        (value_pred_clipped - b_returns).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (value_pred - b_returns).pow(2).mean()

                # 使用按进度调度后的当前熵系数 ent_coef_now
                loss = policy_loss + args.vf_coef * value_loss - ent_coef_now * entropy.mean()
                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    approx_kl = (b_logprob - new_logprob).mean().abs()
                    clip_fraction += (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.mean().item()
                approx_kl_total += approx_kl.item()
                nbatches += 1

        buffer.clear()
        global_step += args.num_steps * args.batch_size
        avg_reward = reward_trackers["reward"] / max(1, step_count)
        avg_distance = reward_trackers["distance"] / max(1, step_count)
        collision_rate = collision_count / max(1, step_count * args.batch_size)
        success_rate = success_count / max(1, step_count * args.batch_size)

        # 按 episode 统计成功/碰撞率：这一轮 rollout 里，有多少条 episode 是以成功/碰撞结束
        episodes_total = max(1, episodes_success + episodes_collision + episodes_other)
        episode_success_rate = episodes_success / episodes_total
        episode_collision_rate = episodes_collision / episodes_total

        if writer:
            writer.add_scalar("charts/avg_reward", avg_reward, global_step)
            #writer.add_scalar("charts/avg_distance_to_obstacle", avg_distance, global_step)
            writer.add_scalar("charts/collision_rate", collision_rate, global_step)
            writer.add_scalar("charts/success_rate", success_rate, global_step)
            writer.add_scalar("charts/episode_success_rate", episode_success_rate, global_step)
            writer.add_scalar("charts/episode_collision_rate", episode_collision_rate, global_step)
            if recent_returns:
                writer.add_scalar(
                    "charts/episode_return_mean",
                    sum(recent_returns) / len(recent_returns),
                    global_step,
                )
                writer.add_scalar(
                    "charts/episode_length_mean",
                    sum(recent_lengths) / len(recent_lengths),
                    global_step,
                )

            writer.add_scalar("losses/policy", policy_loss_total / max(1, nbatches), global_step)
            writer.add_scalar("losses/value", value_loss_total / max(1, nbatches), global_step)
            writer.add_scalar("losses/entropy", entropy_total / max(1, nbatches), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl_total / max(1, nbatches), global_step)
            writer.add_scalar("losses/clip_fraction", clip_fraction / max(1, nbatches), global_step)

            # 记录当前熵系数，方便在 TensorBoard 中观察其调度曲线
            writer.add_scalar("components/ent_coef", ent_coef_now, global_step)

            # 把各个观测项乘上它们在 reward 中的系数，方便直接看到“对 reward 的贡献量级”
            denom = max(1, step_count)
            # writer.add_scalar(
            #     "components/loss_v",
            #     args.coef_v * reward_trackers["loss_v"] / denom,
            #     global_step,
            # )
            writer.add_scalar(
                "components/loss_obj",
                args.coef_obj_avoidance * reward_trackers["loss_obj"] / denom,
                global_step,
            )
            writer.add_scalar(
                "components/loss_collide",
                args.coef_collide * reward_trackers["loss_collide"] / denom,
                global_step,
            )
            writer.add_scalar(
                "components/loss_d_acc",
                args.coef_d_acc * reward_trackers["loss_d_acc"] / denom,
                global_step,
            )
            writer.add_scalar(
                "components/loss_d_jerk",
                args.coef_d_jerk * reward_trackers["loss_d_jerk"] / denom,
                global_step,
            )
            # writer.add_scalar(
            #     "components/loss_d_snap", 
            #     args.coef_d_snap * reward_trackers["loss_d_snap"] / denom,
            #     global_step,
            # )
            # writer.add_scalar(
            #     "components/loss_alive",
            #     reward_trackers["loss_alive"] / denom,
            #     global_step,
            # )
            writer.add_scalar(
                "components/loss_bias",
                args.coef_bias * reward_trackers["loss_bias"] / denom,
                global_step,
            )
            # writer.add_scalar(
            #     "components/loss_speed",
            #     args.coef_speed * reward_trackers["loss_speed"] / denom,
            #     global_step,
            # )
            # writer.add_scalar(
            #     "components/loss_v_pred",
            #     args.coef_v_pred * reward_trackers["loss_v_pred"] / denom,
            #     global_step,
            # )
            writer.add_scalar(
                "components/progress",
                args.progress_coef * reward_trackers["progress"] / denom,
                global_step,
            )
            writer.add_scalar(
                "components/goal_bonus",
                reward_trackers["goal_bonus"] / denom,
                global_step,
            )

        if args.save_interval > 0 and (update + 1) % args.save_interval == 0:
            save_path = os.path.join(args.log_dir, f"ppo_checkpoint_{update+1:05d}.pth")
            torch.save(model.state_dict(), save_path)

    writer.close()


if __name__ == "__main__":
    main()
