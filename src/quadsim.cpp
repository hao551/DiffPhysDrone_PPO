#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


// quadsim.cpp
//  中的是函数声明（前向声明）
// quadsim_kernel.cu
//  中有两层实现：
// C++接口函数：管理CUDA调用和类型分发
// CUDA核函数：实际的GPU并行计算逻辑
void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

// 计算深度图对无人机位置的梯度，使得渲染过程可以参与神经网络的反向传播
// 网络可以学习"看哪里"
// 例如：避障时主动调整视角
// 可微分方法：主动寻找最佳观察角度；传统方法：被动接受当前视角
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

    
// 碰撞检测和避障
// 找到离无人机最近的障碍物点
// 用于计算安全距离和避障损失
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);
    
// 作用：姿态更新
// 根据推力方向和速度预测更新无人机姿态
// 计算新的旋转矩阵（前向、左向、上向轴）
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

// 物理仿真前向传播
// CUDA加速的物理状态更新
// 集成推力、阻力、扰动等所有物理效应
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
    float airmode_av2a);

// 物理仿真反向传播
// 计算物理仿真过程的梯度
// 支持端到端训练中的梯度传播
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
    float ctl_dt);

// C++ interface

// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render", &render_cuda, "render (CUDA)");
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");
}
