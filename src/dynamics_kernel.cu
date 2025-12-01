#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>//这个函数不是固定写死 float 或 double，而是可以生成多个版本
//__global__ 是 CUDA 的关键字，表示：这是一个 CUDA kernel，运行在 GPU 上，由 CPU 启动,必须用 <<<blocks, threads>>> 调用
//在 GPU 上并行执行返回必须是 void（不能 return）
//这个函数实现了无人机姿态矩阵的更新，根据推力方向和速度预测计算新的旋转矩阵。
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,//输出：新的姿态矩阵,[B, 3, 3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,//输入：当前的姿态矩阵,[B, 3, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_thr,//输入：推力方向,[B, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_pred,//输入：速度预测,[B, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> alpha,//输入：alpha,[B, 1]
    float yaw_inertia) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;//当前线程处理的无人机索引，这是 CUDA 里计算全局线程 ID 的标准写法。
    const int B = R.size(0);//总的无人机数量（批次大小）
    if (b >= B) return;
    // a_thr = a_thr - self.g_std;a_thr：网络输出的推力加速度（已经减去了重力）
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665;//重新加上重力，得到总推力方向
    // thrust = torch.norm(a_thr, 2, -1, True);
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);//总推力大小
    // self.up_vec = a_thr / thrust;
    //u 这是一个单位向量，表示**“机体的上向（up）方向在世界坐标中的方向”**
    scalar_t ux = ax / thrust;
    scalar_t uy = ay / thrust;
    scalar_t uz = az / thrust;
    // forward_vec = self.forward_vec * yaw_inertia + v_pred;
    // 前向轴 forward_old（机体系 x 轴在世界坐标中的方向）
    // R[b][0][0]：前向轴在 world-x 的分量
    // R[b][1][0]：前向轴在 world-y 的分量
    // R[b][2][0]：前向轴在 world-z 的分量
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    // forward_vec = F.normalize(forward_vec, 2, -1);
    // forward_vec = (1-alpha) * forward_vec + alpha * self.forward_vec
    //用 alpha 做指数平滑（避免抖动）
    scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];
    // forward_vec[2] = (forward_vec[0] * self_up_vec[0] + forward_vec[1] * self_up_vec[1]) / -self_up_vec[2]
    fz = (fx * ux + fy * uy) / -uz;//确保前向轴垂直于上向轴
    // self.forward_vec = F.normalize(forward_vec, 2, -1);
    //现在 (fx, fy, fz) 是一个单位向量，而且满足：单位长度与 up_vec 正交，所以它是新的前向方向。
    t = sqrt(fx * fx + fy * fy + fz * fz);
    fx /= t;
    fy /= t;
    fz /= t;
    // self.left_vec = torch.cross(self.up_vec, self.forward_vec);
    //用up 和 forward 构造第三个轴：left = up × forward
    R_new[b][0][0] = fx;
    R_new[b][0][1] = uy * fz - uz * fy;
    R_new[b][0][2] = ux;
    R_new[b][1][0] = fy;
    R_new[b][1][1] = uz * fx - ux * fz;
    R_new[b][1][2] = uy;
    R_new[b][2][0] = fz;
    R_new[b][2][1] = ux * fy - uy * fx;
    R_new[b][2][2] = uz;
}

//这个函数实现了一个时间步的物理状态更新，包括推力控制、阻力计算、运动学积分等
template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,//输入：当前的姿态矩阵,[B, 3, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,// [B,3] 随机扰动力
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,// [B,1] z 方向阻力系数
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,// [B,1] 阻力系数
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,// [B,1] 俯仰控制延迟
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_pred,// [B,3] 预测的动作
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act,// [B,3] 当前动作
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,// [B,3] 当前位置
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,// [B,3] 当前速度
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,// [B,3] 当前风速
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,// [B,3] 当前加速度
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,// [B,3] 下一时刻的动作
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,// [B,3] 下一时刻的位置
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,// [B,3] 下一时刻的速度
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,// [B,3] 下一时刻的加速度
    float ctl_dt, float airmode_av2a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;//i 决定：当前这个 GPU 线程负责第 i 个样本（第 i 架无人机）。
    const int B = R.size(0);
    if (i >= B) return;
    //推力控制延迟（一阶低通滤波）
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);//控制延迟：模拟电机响应延迟
    // self.act = act_pred * (1 - alpha) + self.act * alpha
    for (int j=0; j<3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    // self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
    // v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
    //// 相对风速
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    //// 投影到机体坐标系的三个轴
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    // scalar_t v_up[3]
    // for (int j=0; j<3; j++){
    //     v_up[j] = v_up_s * R[i][j][2];
    // }
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    //计算二次阻力项
    //二次阻力：阻力与速度的平方成正比
    //v * |v| 保留符号，方向与速度相反
    scalar_t v_up_2 = v_up_s * abs(v_up_s);
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
    scalar_t v_left_2 = v_left_s * abs(v_left_s);

    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j=0; j<3; j++){
        //二次阻力（a_drag_2）
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        //一次阻力（a_drag_1）
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }
    // v_prep = self.v - v_up
    // scalar_t v_prep[3];
    // for (int j=0; j<3; j++)
    //     v_prep[j] = v[i][j] - v_up[j];
    // motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
    // 计算推力方向变化的角速度
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    // 角速度 = arccos(夹角) / dt
    scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;

    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a};
    // scalar_t motor_velocity = sqrt(sqrt(act_x * act_x + act_y * act_y + act_z * act_z));
    // z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
    // a_next = self.act + self.dg - z_drag
    // scalar_t v_scalar = sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
    //总加速度计算
    for (int j=0; j<3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
    
    //这是匀加速运动的位移公式:p_next = p + v * dt + 0.5 * a * dt²
    for (int j=0; j<3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    //v_next = v + (a + a_next) / 2 * dt,这个是二阶泰勒展开
    for (int j=0; j<3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}


//这是物理仿真反向传播的核心函数，实现了前向传播的自动微分
template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,// [B,3,3] 姿态矩阵
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,// [B,3] 随机扰动力
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,// [B,1] z 方向阻力系数
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,// [B,1] 阻力系数
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,// [B,1] 俯仰控制延迟
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,// [B,3] 当前速度
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,// [B,3] 当前风速
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,// [B,3] 下一时刻的动作
    // 输出：对输入的梯度
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act_pred,// [B,3] 预测的动作的梯度∂L/∂act_pred
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act,// [B,3] 当前动作的梯度∂L/∂act
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,// [B,3] 当前位置的梯度∂L/∂p
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,// [B,3] 当前速度的梯度∂L/∂v
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,// [B,3] 当前加速度的梯度∂L/∂a
    // 这个函数输入：对输出的梯度，是 PyTorch 自动传入的
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_act_next,// [B,3] 下一时刻的动作的梯度∂L/∂act_next
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,// [B,3] 下一时刻的位置的梯度∂L/∂p_next
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,// [B,3] 下一时刻的速度的梯度∂L/∂v_next
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,// [B,3] 下一时刻的加速度的梯度∂L/∂a_next
    float grad_decay,// 梯度衰减因子
    float ctl_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    // // self.act = act_pred * (1 - alpha) + self.act * alpha
    // for (int j=0; j<3; j++)
    //     act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    // // self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
    // // v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
    // scalar_t v_up_s = v[i][0] * R[i][0][2] + v[i][1] * R[i][1][2] + v[i][2] * R[i][2][2];
    // scalar_t v_up[3];
    // for (int j=0; j<3; j++)
    //     v_up[j] = v_up_s * R[i][j][2];
    // // v_prep = self.v - v_up
    // scalar_t v_prep[3];
    // for (int j=0; j<3; j++)
    //     v_prep[j] = v[i][j] - v_up[j];
    // motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
    scalar_t act_x = act_next[i][0];
    scalar_t act_y = act_next[i][1];
    scalar_t act_z = act_next[i][2] + 9.80665;
    // scalar_t motor_velocity = sqrt(sqrt(act_x * act_x + act_y * act_y + act_z * act_z));
    // // z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
    // // a_next = self.act + self.dg - z_drag
    // y = ax; dy/dx = a; dy = a dx
    // scalar_t v_scalar = sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
    // for (int j=0; j<3; j++)
    //     a_next[i][j] = act_next[i][j] + dg[i][j] - z_drag_coef[i][0] * v_prep[j] * motor_velocity * 0.07 - drag_2[i][0] * v_scalar * v[i][j];
    // // self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
    // for (int j=0; j<3; j++)
    //     p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    // // self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
    // for (int j=0; j<3; j++)
    //     v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
    
    // 复制输入梯度到局部变量
    scalar_t d_act_next[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    // backward starts 
    //注意这些都是对loss求微分
    for (int j=0; j<3; j++){
        // v_next[i][j] = v[i][j]* * (grad_decay ** ctl_dt) + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];
        d_a_next[j] += 0.5 * ctl_dt * d_v_next[i][j];
    }
    for (int j=0; j<3; j++){
        // p_next[i][j] = p[i][j] * (grad_decay ** ctl_dt) + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt);
        d_v[i][j] += ctl_dt * d_p_next[i][j];
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j];
    }
    // scalar_t d_v_prep[3] = {0, 0, 0};
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    // scalar_t d_v_scalar = 0;
    for (int j=0; j<3; j++){
        // a_next[i][j] = act_next[i][j] + dg[i][j] - z_drag_coef[i][0] * v_prep[j] * motor_velocity * 0.07 - a_drag_2 - a_drag_1;
        d_act_next[j] += d_a_next[j];
        // d_v_prep[j] -= z_drag_coef[i][0] * d_a_next[j] * motor_velocity * 0.07;
        // d_v_scalar -= d_a_next[j] * drag_2[i][0] * v[i][j];
        // d_v[i][j] -= d_a_next[j] * drag_2[i][0] * v_scalar;
        d_a_drag_2[j] = -d_a_next[j] * drag_2[i][0];
        d_a_drag_1[j] = -d_a_next[j] * drag_2[i][1];
    }
    // for (int j=0; j<3; j++)
    //     d_v[i][j] += d_v_scalar * v[i][j] / v_scalar;

    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    for (int j=0; j<3; j++){
        // a_drag_2[j] = v_up_s * v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * v_left_s * R[i][j][1] + v_fwd_s * v_fwd_s * R[i][j][0];
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1];
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0];
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0];
    }

    // scalar_t d_v_up[3] = {0, 0, 0};
    // for (int j=0; j<3; j++){
    //     // v_prep[j] = v[i][j] - v_up[j];
    //     d_v[i][j] += d_v_prep[j];
    //     d_v_up[j] -= d_v_prep[j];
    // }
    // for (int j=0; j<3; j++){
    //     // v_up[j] = v_up_s * R[i][j][2];
    //     d_v_up_s += d_v_up[j] * R[i][j][2];
    // }
    // scalar_t v_up_s = v[i][0] * R[i][0][2] + v[i][1] * R[i][1][2] + v[i][2] * R[i][2][2];
    //坐标转换的反向传播
    for (int j=0; j<3; j++){
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;
        d_v[i][j] += R[i][j][1] * d_v_left_s;
        d_v[i][j] += R[i][j][2] * d_v_up_s;
    }
    //坐标转换的反向传播
    for (int j=0; j<3; j++){
        // act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
        d_act_pred[i][j] = (1 - alpha) * d_act_next[j];
        d_act[i][j] = alpha * d_act_next[j];
    }
}

} // namespace 匿名 。namespace 内：实现细节（CUDA kernels），匿名 namespace 外：公开接口（C++ 包装函数）

// 返回 4 个 Tensor 的向量
// [act_next, p_next, v_next, a_next]
// run_forward_cuda_kernel - CUDA kernel（模板函数）
// run_forward_cuda - C++ 包装函数
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a){// 空中模式参数

    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    const int threads = R.size(0);
    const dim3 blocks(1);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt, airmode_av2a);
    }));
    return {act_next, p_next, v_next, a_next};
}

std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt){

    torch::Tensor d_act_pred = torch::empty_like(dg);
    torch::Tensor d_act = torch::empty_like(dg);
    torch::Tensor d_p = torch::empty_like(dg);
    torch::Tensor d_v = torch::empty_like(dg);
    torch::Tensor d_a = torch::empty_like(dg);

    const int threads = R.size(0);
    const dim3 blocks(1);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, ctl_dt);
    }));
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    const int threads = a_thr.size(0);
    const dim3 blocks(1);
    torch::Tensor R_new = torch::empty_like(R);
    AT_DISPATCH_FLOATING_TYPES(a_thr.scalar_type(), "update_state_vec", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            a_thr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            yaw_inertia);
    }));
    return R_new;
}
