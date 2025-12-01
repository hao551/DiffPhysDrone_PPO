## 一、总体目标与约束

- **目标**：在不修改现有代码的前提下，新建一套基于 **PPO** 的强化学习训练脚本，用于在当前 CUDA 物理仿真环境中重新学习无人机导航策略。  
- **复用要求**：  
  - 继续使用 `env_cuda.Env` 中的 **CUDA 物理仿真**（`render`, `run`, `find_vec_to_nearest_pt` 等）。  
  - **深度图处理逻辑保持与 `main_cuda.py` 一致**（`render → 3/depth 变换 → max_pool2d` 等）。  
  - **奖励/惩罚设计尽量沿用原项目的损失设计思想**，本质上把原来的“损失项”改写为 RL 的“即时奖励”。  
- **约束**：  
  - 不改动任何已有文件（`env_cuda.py`, `main_cuda.py`, `model.py` 等保持原样）。  
  - 新增 PPO 相关文件，形成一个相对独立但依赖原环境的 RL 子项目。

---

## 二、现有代码的功能拆解（为 PPO 设计做准备）

### 2.1 环境与物理仿真（`Env`）

- **环境类**：`Env(batch_size, width, height, grad_decay, device, ...)`  
  - 内部维护障碍物（球、立方体、圆柱、门、地形等）、无人机状态（位置 `p`、速度 `v`、姿态 `R`、加速度 `a`、风 `v_wind`、控制量 `act` 等）。  
- **关键接口**：
  - `reset()`：重新随机化场景、重置无人机状态、目标点 `p_target` 等（当前训练中每个迭代都会 `env.reset()`）。  
  - `render(ctl_dt)`：  
    - 调用 CUDA 内核生成深度图 `depth`（形状 `[B, H, W]`）。  
    - 使用相机姿态 `R_cam`、无人机姿态 `R`、障碍物参数等。  
  - `run(act_pred, ctl_dt, v_pred=None)`：  
    - 使用 CUDA 内核 `run_forward` 演化物理状态，更新 `p, v, a, R` 等。  
    - 这里的 `act_pred` 是“控制指令”而非网络原始输出，当前由 `main_cuda.py` 根据网络输出和重力、估计误差组合得到。  
  - `find_vec_to_nearest_pt()`：  
    - 对每架无人机，计算到最近障碍物表面的向量（用于避障损失/奖励）。  

### 2.2 感知与状态构造（`main_cuda.py`）

在当前 supervised / imitation 风格训练脚本中，每个时间步主要做：

- **深度图处理**：
  - `depth, flow = env.render(ctl_dt)`  
  - `x = 3 / depth.clamp(0.3, 24) - 0.6 + 噪声`  
  - `x = F.max_pool2d(x[:, None], 4, 4)` → 输入模型的单通道低分辨率深度图。  
- **状态向量 `state`**（无 `--no_odom` 时）：
  - `local_v`：当前速度在简化机体系（只保留偏航）的坐标下表示。  
  - `target_v_body`：目标速度（从 `p` 指向 `p_target`，再截断到 `max_speed`）投影到简化机体系下。  
  - `env.R[:, 2]`：机体“上向轴”，表示姿态信息。  
  - `env.margin`：安全裕度（与障碍物距离的 buffer）。  
  - 合并后维度约为 10（默认 `Model(7+3, 6)`）。  

### 2.3 模型结构与动作定义（`model.py`）

- **主干网络**：
  - 深度图经过 3 层 CNN + Flatten + 全连接 → 192 维视觉特征。  
  - 状态向量 `v` 经过 `Linear(dim_obs → 192)` → 192 维状态特征。  
  - 二者相加、激活后送入 `GRUCell(192 → 192)`。  
- **输出**：
  - 线性层输出 `dim_action` 维（在 `main_cuda.py` 中为 6 维）。  
  - 在 `main_cuda.py` 中使用方式：  
    - 将 `act` reshape 为 `[B, 3, 2]`，乘以（简化的）机体系旋转矩阵 `R`，  
    - 得到两个 3 维向量：`a_pred`（期望加速度）和 `v_pred`（速度估计）。  

### 2.4 现有“损失项”可转化为“奖励项”

当前脚本中定义了一系列损失：

- **轨迹跟踪相关**：
  - `loss_v`：真实速度与目标速度之间的 Smooth L1 差（通过 30 步滑动平均）。  
  - `loss_v_pred`：网络估计速度 `v_pred` 与真实速度 `v` 的 MSE，用于“无里程计”辅助学习。  
  - `loss_bias`：真实速度与目标方向对齐程度的偏差。  
  - `loss_speed`：速度模长与目标速度模长的偏差。  
- **安全/避障相关**：
  - `loss_obj_avoidance`：通过 barrier 函数，鼓励远离障碍物（基于与障碍物距离及距离变化率）。  
  - `loss_collide`：基于距离的 Softplus 惩罚，模拟“接近/碰撞”高惩罚。  
  - `loss_ground_affinity`：z 轴高度为负（撞地/穿地）时的惩罚。  
- **控制平滑性**：
  - `loss_d_acc`：推力大小平方和。  
  - `loss_d_jerk`：一阶差分（jerk）平方和。  
  - `loss_d_snap`：高阶差分（snap）平方和。  

在 PPO 中，这些可以被视为 **代价项**，每步即时奖励可以设计为它们的负线性组合。

---

## 三、新 PPO 子项目的模块划分（新增文件）

### 3.1 新增文件结构（建议）

- **`ppo_model.py`**  
  - 定义 PPO 用的 **Actor-Critic 网络**，复用现有 CNN + GRU 主干结构。  
- **`ppo_buffer.py`**  
  - 实现 Rollout Buffer，用于存储 `obs, state, action, logprob, value, reward, done`，并支持 GAE 计算与 mini-batch 采样。  
- **`ppo_trainer.py`**（或 `train_ppo.py`）  
  - 训练主脚本：环境创建、采样 rollout、计算优势、PPO 更新、日志与模型保存。  
- **`ppo_utils.py`**（可选）  
  - 公共工具函数：优势/回报计算、学习率调度、归一化工具等。  
- **`configs/ppo.args`**（可选）  
  - 独立于原项目的 PPO 超参数配置（也可以直接用 argparse 写在 `ppo_trainer.py` 里）。

### 3.2 与原项目的依赖关系

- 所有 PPO 代码只通过 **import** 方式使用现有模块：  
  - `from env_cuda import Env`  
  - 深度与状态构造逻辑直接复用 `main_cuda.py` 的写法（可以在 PPO 内部按相同公式写，而不是修改原脚本）。  
- 不修改原有文件，只是共享：  
  - CUDA 动态库 `quadsim_cuda`（间接通过 `Env`）  
  - 物理仿真与渲染接口  
  - 状态/深度的计算配方（代码逻辑在新文件中重新实现一遍）。

---

## 四、PPO 模型与数据流设计

### 4.1 Actor-Critic 模型（`ppo_model.PPOModel`）

- **主干结构**：  
  - 复制 `Model` 的 `stem`（CNN）与 `v_proj`（状态投影）、`gru`（GRUCell）。  
- **Head 设计**：
  - **Actor head**：输出高斯策略的均值向量 `mu`，维度为 6（对应 `[a_pred_body(3), v_pred_body(3)]`）。  
    - 另有可学习的 `log_std` 向量（6 维），或固定超参数。  
    - 最终策略为 `π(a | s) = N(mu, diag(exp(2*log_std)))`。  
  - **Critic head**：输出标量状态价值 `V(s)`。  
- **前向接口**：  
  - 输入：`depth_tensor [B, 1, H', W']`，`state_vector [B, dim_obs]`，GRU 隐状态 `hx`。  
  - 输出：`action_mu [B, 6]`, `value [B, 1]`, `log_std [6] or [B, 6]`, 更新后的 `hx`。  

### 4.2 观测与动作在 PPO 中的流转

- **观测**：
  - **深度部分**：完全照抄 `main_cuda.py` 的处理流程。  
  - **状态部分**：  
    - 是否包含 `local_v` 由 PPO 的 `--no_odom` 参数控制，逻辑与原项目相同。  
    - 其他字段（`target_v_body`, `env.R[:, 2]`, `env.margin`）保持一致。  
- **动作**：
  - PPO 直接输出 6 维连续动作 `a_t`（actor 的采样结果）：  
    - 通过 reshape+旋转解读为：`a_pred_body`, `v_pred_body`。  
    - 再通过现有的组合公式：  
      - 在世界系计算期望推力：`a_pred_world`，`v_pred_world`。  
      - 用 `env.g_std` 和 `env.thr_est_error` 组合为最终控制量 `act`，传给 `env.run(act, ctl_dt, target_v_raw)`。  
  - 如此即可**保证物理接口与当前训练脚本一致**。  

---

## 五、奖励函数设计（沿用原损失思想）

### 5.1 基本思路

- 将原脚本中的每个损失项视为“每步代价”的构成部分。  
- PPO 中的即时奖励定义为：  

\[
r_t = -(
c_v \cdot \ell_v^t +
c_{\text{avoid}} \cdot \ell_{\text{avoid}}^t +
c_{\text{collide}} \cdot \ell_{\text{collide}}^t +
c_{\text{smooth}} \cdot \ell_{\text{smooth}}^t +
c_{\text{ground}} \cdot \ell_{\text{ground}}^t
)
\]

- 其中各个 \(\ell^t\) 分别为当前时间步（或与附近时间步构造）上的对应损失项，\(c_*\) 为可调权重。

### 5.2 建议的奖励分解

在 PPO rollout 中，先按时间收集必要的历史量（与 `main_cuda.py` 类似），但最后不做整体加权求和，而是为每个时间步构造 `r_t`：

- **目标速度/方向奖励**：
  - 使用 `v_history`, `target_v_history` 构造：  
    - `loss_v^t`：当前或短窗口的速度偏差。  
    - `loss_bias^t`：速度方向与目标方向的偏差。  
    - `loss_speed^t`：速度模长偏差。  
  - 奖励项：  
    - \(r^{(v)}_t = - (w_v \ell_v^t + w_{\text{bias}} \ell_{\text{bias}}^t + w_{\text{speed}} \ell_{\text{speed}}^t)\)。  
- **避障与碰撞惩罚**：
  - 利用 `vec_to_pt_history` 计算即刻距离 `distance_t` 及其差分。  
  - `loss_obj_avoidance^t`、`loss_collide^t` 的构造与原逻辑类似，只是在时间上对齐到第 `t` 步。  
  - 奖励项：  
    - 若 `distance_t < 0`（安全裕度内），施加强惩罚；  
    - 若 `distance_t` 增大（远离障碍物），给予少量正奖励。  
- **姿态与高度安全**：
  - 继承 `loss_ground_affinity`，对 `p[...,2]` 小于 0 的程度进行惩罚：  
    - \(r^{(ground)}_t = - w_{ground} \ell^{ground}_t\)。  
- **控制平滑性**：
  - 一阶差分 `jerk_history^t` 和高阶差分 `snap_history^t`，与原逻辑相同。  
  - 奖励项：  
    - \(r^{(smooth)}_t = - (w_{acc} \ell_{acc}^t + w_{jerk} \ell_{jerk}^t + w_{snap} \ell_{snap}^t)\)。  
- **终止/成功奖励（可选增强）**：
  - 在 rollout 末尾，用当前与目标点距离 `||p - p_target||` 判断是否达到任务目标：  
    - 若距离小于阈值且整段轨迹无碰撞，可给予一次性“大额正奖励”。  

---

## 六、PPO 训练流程设计（算法层）

### 6.1 超参数建议

- 环境与并行：
  - `num_envs` = `batch_size`（如 64），通过 `Env(batch_size=...)` 实现多并行轨迹。  
  - 每个 rollout `num_steps`（如 128–256）与原来的 `timesteps` 保持同量级。  
- PPO：
  - 折扣因子 `gamma ≈ 0.99`，GAE 参数 `lambda ≈ 0.95`。  
  - 每次迭代更新轮数 `ppo_epochs ≈ 4–10`。  
  - 每次迭代的 mini-batch 大小 `batch_size = num_envs * num_steps / n_minibatches`。  
  - Clip 范围 `clip_range ≈ 0.1–0.2`，价值函数 Clip 同级别。  
  - 学习率 `lr ≈ 3e-4`，可采用 Cosine 或 Linear decay。  

### 6.2 采样与更新的伪流程（`ppo_trainer.py`）

1. **初始化**：
   - 解析 PPO 的命令行参数（可参考 `main_cuda.py` 的模式）。  
   - 创建 `Env`、`PPOModel`、`RolloutBuffer`、优化器与学习率调度器。  
2. **主训练循环（for update in range(num_updates)）**：
   1. `env.reset()`、`model.reset()`，清空 GRU 隐状态。  
   2. 对每个时间步 `t in range(num_steps)`：  
      - 采样随机 `ctl_dt`（与原脚本一样）。  
      - `depth, _ = env.render(ctl_dt)`，构造深度观测 `x`。  
      - 构造状态向量 `state`（含/不含 `local_v` 等）。  
      - 前向通过 `PPOModel` 得到 `mu, value, log_std, hx`。  
      - 从策略分布中采样动作 `a_t`，计算 `logprob_t`。  
      - 将 `a_t` 转成 `act`，调用 `env.run(act, ctl_dt, target_v_raw)`（逻辑与原脚本一致）。  
      - 调用 `env.find_vec_to_nearest_pt()` 等，记录构造奖励所需信息。  
      - 根据第 5 节所述公式，计算即时奖励 `r_t` 和 `done_t`（可用固定长度回合，`done=False`，或引入碰撞终止）。  
      - 将 `obs, state, action, logprob, value, reward, done` 推入 `RolloutBuffer`。  
   3. **回报与优势计算**：  
      - 使用最后一个时间步的价值估计 `V(s_T)` 作为 bootstrap。  
      - 在 `ppo_buffer` 内部按 GAE 计算 `adv_t` 与折扣回报 `R_t`。  
   4. **PPO 更新**：  
      - 对 buffer 中的数据打乱，按 mini-batch 多轮更新：  
        - 再次前向得到新 `logprob` 与 `value`。  
        - 计算 `ratio = exp(new_logprob - old_logprob)`。  
        - 采用 clipped surrogate loss 计算 policy loss。  
        - 价值损失用 MSE 或 Clipped MSE。  
        - 可加一个 entropy bonus，鼓励探索。  
        - 再加（可选）辅助 loss：如 `loss_v_pred`，继续训练速度预测头，以帮助稳定训练。  
      - 反向传播、梯度裁剪、优化器更新、学习率调度。  
3. **日志与可视化**：
   - 复用 TensorBoard（`SummaryWriter`），记录：  
     - reward、success rate、平均速度、最近障碍距离等统计量。  
     - 选取某个 env 的深度图序列写入 video，类似 `main_cuda.py` 的 `writer.add_video`。  
4. **模型保存**：
   - 定期保存 `PPOModel` 的权重 checkpoint（命名如 `ppo_checkpoint_xxxx.pth`）。  

---

## 七、配置与运行方式设计

### 7.1 命令行参数（示例）

在 `ppo_trainer.py` 中使用 `argparse`，推荐参数包括：

- 环境相关：`--batch_size`, `--timesteps`, `--grad_decay`, `--speed_mtp`, `--fov_x_half_tan`, `--cam_angle`, `--single`, `--gate`, `--ground_voxels`, `--scaffold`, `--random_rotation`, `--yaw_drift`, `--no_odom`（与原脚本保持一致）。  
- PPO 相关：  
  - `--num_updates`, `--num_steps`, `--gamma`, `--gae_lambda`,  
  - `--ppo_epochs`, `--num_minibatches`, `--clip_range`,  
  - `--vf_coef`, `--ent_coef`, `--max_grad_norm`,  
  - `--lr`, `--lr_schedule`。  
- 日志与 checkpoint：`--log_dir`, `--save_interval`, `--resume` 等。  

### 7.2 与原项目并存方式

- 通过独立入口 `python ppo_trainer.py --...` 启动，不影响 `main_cuda.py`。  
- 共用 `runs/` 或另开 `runs_ppo/` 目录存放 TensorBoard 日志与权重。  
- 所有新文件仅 import 现有模块，无需改动现有任何源码。

---

## 八、后续扩展方向（可选）

- 在 PPO 中引入 **多任务 reward**：如加入通过门任务、编队任务，进一步利用 `Env` 的参数（如 `gate`, `scaffold` 等）。  
- 把 `Env` 封装成 `gymnasium` 风格环境（实现 `reset/step` 接口），便于未来迁移到通用 RL 框架（Stable-Baselines3 等）——仍然不改原 `Env`，而是新建 wrapper。  
- 在 PPO 中加入 **curriculum learning**：逐步增大障碍密度、风速、目标距离。  
 
---

## 九、碰撞后的“局部重置”策略设计（只重置撞到的那几个）

### 9.1 设计动机

- 默认做法是：**整批 reset**（所有无人机统一 `env.reset()`），简单但浪费尚未撞击的轨迹。  
- 更高效的做法是：**只为“撞到障碍物”的那些 index 重新采样初始状态**，其他仍继续当前 episode。  
- 在不修改 `Env` 源码的前提下，可以通过 **“影子环境 + 状态拷贝”** 的方式实现局部重置。

### 9.2 碰撞判定（回顾）

```python
vec_to_pt = env.find_vec_to_nearest_pt()    # [B, 3]
distance = vec_to_pt.norm(2, -1)            # [B]
distance_for_loss = distance - env.margin   # 与原损失一致
collided = distance_for_loss < 0            # 或者 distance < env.drone_radius
```

- `collided` 是形状为 `[B]` 的布尔张量，表示当前时间步哪些无人机视为“碰撞/极度危险”。  
- 在 PPO 中，这些 index 的样本：  
  - 即刻奖励中加上大的惩罚 `-collision_penalty`；  
  - 把 `done[i] = True`，在 GAE 计算时视作 episode 结束。  

### 9.3 影子环境 + 状态拷贝的局部重置方案

核心思路：新建一个 **影子环境** `env_template = Env(...)`，当某个 index 撞到障碍物时，从 `env_template` 中取一份“新样本”的状态，覆盖当前环境中对应 index 的状态，从而实现“部分 reset”。

关键点：

- **不修改 `Env` 类接口**，完全在 PPO 脚本里操作。  
- 对 `p, v, a, R, act, dg, v_wind, balls, voxels, cyl, cyl_h, p_target, margin, max_speed, pitch_ctl_delay, yaw_ctl_delay` 等多组张量做 indexed 赋值。  
- 每次局部重置前，先对 `env_template.reset()` 一次，保证从其中取到“新场景/新起点”。

示意伪代码（仅说明逻辑，实际实现时需根据 `Env` 内部变量补全字段）：

```python
device = torch.device("cuda")
env = Env(batch_size, 64, 48, args.grad_decay, device, ...)
env_template = Env(batch_size, 64, 48, args.grad_decay, device, ...)

def partial_reset(env, env_template, collided_mask: torch.Tensor):
    """
    只对 collided_mask == True 的那些 index 做局部重置。
    collided_mask: [B] 的 bool 张量。
    """
    if not collided_mask.any():
        return

    # 先在影子环境中重新采样一个 batch
    env_template.reset()

    idx = collided_mask.nonzero(as_tuple=False).squeeze(-1)  # [K]

    # 注意：这里用的是“从 env_template 拿同 index 的样本”，
    # 如果想完全打乱，也可以随机采样 template 的 index。
    env.p[idx] = env_template.p[idx]
    env.v[idx] = env_template.v[idx]
    env.a[idx] = env_template.a[idx]
    env.R[idx] = env_template.R[idx]
    env.R_old[idx] = env_template.R_old[idx]
    env.p_old[idx] = env_template.p_old[idx]
    env.p_target[idx] = env_template.p_target[idx]

    env.act[idx] = env_template.act[idx]
    env.dg[idx] = env_template.dg[idx]
    env.v_wind[idx] = env_template.v_wind[idx]

    env.balls[idx] = env_template.balls[idx]
    env.voxels[idx] = env_template.voxels[idx]
    env.cyl[idx] = env_template.cyl[idx]
    env.cyl_h[idx] = env_template.cyl_h[idx]

    env.margin[idx] = env_template.margin[idx]
    env.max_speed[idx] = env_template.max_speed[idx]
    env.pitch_ctl_delay[idx] = env_template.pitch_ctl_delay[idx]
    env.yaw_ctl_delay[idx] = env_template.yaw_ctl_delay[idx]
```

在 PPO 采样循环中，每一步可以这样使用：

```python
for t in range(num_steps):
    # 1. 渲染、前向网络、采样动作、env.run(...)
    depth, _ = env.render(ctl_dt)
    # ... 构造 obs / state，前向 PPOModel，env.run(act, ctl_dt, target_v_raw)

    # 2. 碰撞检测
    vec_to_pt = env.find_vec_to_nearest_pt()
    distance = vec_to_pt.norm(2, -1)
    distance_for_loss = distance - env.margin
    collided = distance_for_loss < 0

    # 3. 奖励和 done
    reward = base_reward_fn(...)
    reward[collided] -= collision_penalty
    done = done | collided  # 对应轨迹在 RL 视角上结束

    # 4. 把当前步数据写入 PPO buffer
    buffer.add(obs=..., action=..., reward=reward, done=done, value=value, logprob=logprob, ...)

    # 5. 对物理环境做局部 reset（只重置撞到的）
    partial_reset(env, env_template, collided)
```

实现效果：

- 从 **RL 算法视角**：  
  - `done=True` 的时间步会截断 GAE/回报，代表 episode 结束；  
  - 随后同一 index 上出现的状态被视为新 episode 的起点（下一条轨迹），继续采样。  
- 从 **物理仿真视角**：  
  - 只有发生碰撞的无人机会被替换为新的起点和新环境布局；  
  - 没有碰撞的无人机保持原来的状态和障碍物继续飞行，提高采样利用率。

> 这个“只重置撞到的那几个”的策略，在并行 rollouts 场景下通常比整批 reset 更有效率，是推荐的做法。

