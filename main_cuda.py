from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)#速度倍数，控制无人机最大飞行速度
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
args = parser.parse_args()
writer = SummaryWriter()
print(args)

device = torch.device('cuda')

#args.batch_size (默认64)：并行训练的无人机数量
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
if args.no_odom:
    model = Model(7, 6)
else:
    #如果没有禁用里程计，还会额外包含 local_v = torch.squeeze(env.v[:, None] @ R, 1)，即实际速度在机体系下的 3 维观测
    model = Model(7+3, 6)
model = model.to(device)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15


scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

#障碍物相关的损失项，实现软约束惩罚
def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
B = args.batch_size
for i in pbar:
    env.reset()
    #reset()只是为了清空可能的隐状态，不会修改权重
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []#、与障碍物的相对向量
    act_diff_history = []
    v_preds = []#预测速度
    vid = []#视频记录的帧 
    v_net_feats = []#网络特征 
    h = None

    #控制延迟模拟
    act_lag = 1
    #act 是环境 Env 对象的一个成员变量，用来保存当前时刻每架无人机的推力向量（控制指令）
    #把列表 [env.act] 复制 (act_lag + 1) 份，并串联成一个新列表，所以得到的 act_buffer 里初始时包含了多份同一个 env.act 引用。
    #这样就能一次性构造出长度为 act_lag + 1 的缓冲区，用于后续的控制延迟模拟。
    #act_buffer = [env.act] * (act_lag + 1) 先放入两份当前的动作（因为 act_lag = 1），这样列表长度为 2
    act_buffer = [env.act] * (act_lag + 1)
    #先计算当前位置到目标位置的向量，作为原始目标速度，用于后续归一化和控制器
    target_v_raw = env.p_target - env.p
    #可选随机偏航漂移
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)


    #timesteps（时间步）= 环境交互次数
    #epoch:收集一批交互轨迹（例如 2048 timesteps）用这批数据更新 Actor/Critic 网络多次
    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)#控制周期围绕 15 Hz 随机抖动，模拟感知/控制延迟。
        depth, flow = env.render(ctl_dt)
        #记录当前的全球位置 env.p、与最近障碍物的向量
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        #取第 5 架无人机（索引 4）的深度帧存入 vid。
        if is_save_iter(i):
            vid.append(depth[4])

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
            #使用带滞后（act_buffer）的控制指令推进模拟，更新环境状态（位置、速度、姿态等）
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        #姿态矩阵 env.R，env.R：物理真实姿态（复杂，包含所有旋转
        R = env.R
        #从 [B, 3, 3] 中取第0列 → [B, 3]
        #提取：从每个3×3矩阵中提取第0列的3个元素
        #存储：这3个元素在张量中以行向量形式存储
        #批次：对B个无人机重复此操作，得到B行数据
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        #shape == (N, 3),取消z轴
        fwd[:, 2] = 0## 去除俯仰，只保留偏航
        up[:, 2] = 1  # 固定上向为世界垂直方向
        #fwd 在最后一个维度上做 L₂（Euclid）范数归一化，
        fwd = F.normalize(fwd, 2, -1)
        #再与固定的上向量 [0,0,1] 重新正交化，得到新的机体系旋转矩阵 R
        #世界系 → 机体系的旋转矩阵。重计算的 R：控制友好姿态（简化，只保留偏航）
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        #计算目标速度模长与单位向量，并按最大速度 env.max_speed 做截断，得到期望速度 target_v
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        #这里state为一个list
        state = [
            #target_v 是 [B,3]
            #target_v[:, None] 变成 [B, 1, 3]，行向量
            #R 是旋转矩阵 [B, 3, 3]
            #env.v 是 行向量（shape = [3] 或 [1,3]），因此R写在右边
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None]]#安全裕度 env.margin
        #世界坐标系中的速度投影到无人机自身的局部坐标系
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        #入本机真实速度在机体系下的分量 local_v
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        # normalize
        #depth 先截断到 [0.3, 24]，再做 3/x - 0.6 的对数式变换并加高斯噪声，强化近距离障碍对网络的刺激
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        #单通道深度图压成 1/4 分辨率，既降噪又减少后端卷积算量
        x = F.max_pool2d(x[:, None], 4, 4)
        #更新的 GRU 隐状态 h
        act, values, h = model(x, state, h)

        #act这里搞成列向量的形式，因此左乘R
        #unbind(-1) 会把最后一个维度的 “切片” 拆开。拆成三个tensor
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        #a_pred：网络在机体系中预测的加速度/推力；
        #v_pred：对当前速度的估计，用来抵消误差；
        #env.g_std：重力加速度向量，先减后加相当于做重力补偿；
        #env.thr_est_error[:, None]：每架无人机的推力估计修正因子（标量），用于校正模型输出的幅值。
        #期望推力 = 目标加速度 - 当前速度估计 - 重力
        #把重力重新加回去，得到最终的控制指令
        #因为物理仿真需要的是"总加速度"，而不是净推力
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)

    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    #用累积和技巧计算 30 步滑动平均速度，平滑瞬时抖动。
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds = torch.stack(v_preds)
    #网络输出的速度估计 v_pred 与真实速度做 MSE，用于训练无里程计模式下的速度观测器
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    #惩罚偏离目标方向的侧向/垂直分量，鼓励无人机沿直线飞
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    #fwd_v 是 (B,T) fwd_v[..., None] 变成 (B,T,1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    #act_buffer.diff(1, 0)：沿第 0 维（时间轴）做一阶差分，得到 act[t+1] - act[t]，即推力的变化率（加速度的导数 = jerk，急动度
    #.mul(15) 和 .mul(15**2) 是把离散差分转换成连续时间导数的近似（因为控制周期约为 1/15 秒）。
    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    vec_to_pt_history = torch.stack(vec_to_pt_history)## [T, B, 3]
    distance = torch.norm(vec_to_pt_history, 2, -1)#  # [T, B]
    distance = distance - env.margin#到最近障碍物的距离减去安全裕度
    with torch.no_grad():
        #沿维度 1做差分：distance[t+1] - distance[t]
        #如果无人机靠近障碍物，距离减小，差分为负值
        #如果无人机远离障碍物，距离增大，差分为正值
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    #distance.mul(-32)：距离越小（越接近碰撞），值越大。
    #F.softplus(x) = log(1 + e^x)：平滑的指数函数，当 distance < 0 时急剧增长。
    #同样用 v_to_pt 加权：快速接近时惩罚更重。
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity + loss_ground_affinity

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()


    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})
        log_dict = {}
        if is_save_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            if vid:
                frames = torch.stack(vid).detach()            # (T, H, W)
                frames = frames.clamp(0, 10) / 10              # 归一化到 0-1
                frames = frames.unsqueeze(0).unsqueeze(2)      # (1, T, 1, H, W)
                frames = frames.repeat(1, 1, 3, 1, 1)          # (1, T, 3, H, W)
                writer.add_video('drone_depth_video', frames, i + 1, fps=15)
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            scaler_q.clear()
