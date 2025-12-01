#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void render_cuda_kernel(
    //这里的3表示三个维度的意思
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,//输出：深度图画布,[B, H, W]
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,//输出：光流图,[B, H, W, 2]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,//输入：球体障碍物,[B, N, 4]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,//输入：圆柱体障碍物,[B, N, 3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,//输入：圆柱体障碍物,[B, N, 3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,//输入：体素障碍物,[B, N, 6]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,//输入：当前的姿态矩阵,[B, 3, 3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,//输入：上一时刻的姿态矩阵,[B, 3, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,//输入：当前无人机的位置,[B, 3]
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,//输入：上一时刻无人机的位置,[B, 3]
    float drone_radius,//输入：无人机半径
    int n_drones_per_group,//输入：每组无人机数量
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);
    const int H = canvas.size(1);
    const int W = canvas.size(2);
    //每个线程处理一个像素，总共启动 B×H×W 个线程。
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    //归一化像素坐标
    //dx,dy,dz 就是像素 (u,v) 对应的相机光线在世界坐标系中的方向。
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    //归一化像素坐标
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    scalar_t min_dist = 100;
    scalar_t  t = (-1 - oz) / dz;
    if (t > 0) min_dist = t;

    // others “当前这架无人机的视线，离同组里的其它无人机有多近”
    //计算当前无人机所属分组的第一架无人机的索引。一个向下取整到最近的 n_drones_per_group 的倍数的操作。
    //这个/操作是整数除法，结果会向下取整。比如5 / 3 = 1，7 / 3 = 2。
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    //只看“同一 group 的其它无人机”
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        //把其它无人机当成一个“椭球“
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;
        // (ox + t dx)^2 + (oy + t dy)^2 + 4 (oz + t dz)^2 = r^2
        //这不是球，而是椭球：
        //x、y 方向系数是 1
        //z 方向系数是 4
        //→ 等价于：在“竖直方向 z 上”更“瘦”（撞上更容易），或者说把 z 缩放了 2 倍
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    //这段就是**“相机光线和一堆球体做相交检测”的经典代码，也就是光线打在球上的最近距离**
    for (int i = 0; i < balls.size(1); i++) {
        //球体方程：(x-cx)^2+(y-cy)^2+(z-cz)^2=r^2
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        // 光线起点：(ox, oy, oz) （相机位置/射线起点）
        // 光线方向：(dx, dy, dz)（你刚刚推出来的 ray_dir）
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders 沿 z 轴竖直的柱子（只管 x–y 平面）
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    // cylinders_h 沿 y 轴竖直的柱子（只管 x–z 平面）
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // balls
    //对当前这条光线（相机光线），依次检查它和每一个 voxel（长方体体素）的相交情况，
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];
        scalar_t ry = voxels[batch_base][i][4];
        scalar_t rz = voxels[batch_base][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    //从相机出发沿这个像素方向的光线，遇到所有障碍物（球、椭球、圆柱、voxel）中最近的那个的距离。
    canvas[b][u][v] = min_dist;
}

//这个 CUDA 内核用于计算每架无人机到最近障碍物的向量，这是碰撞检测和避障的核心函数。
template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,// 输出：最近点 [T, B, 3]
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls, // 输入：球体障碍物
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos, // 输入：无人机位置 [T, B, 3]
    float drone_radius, // 输入：无人机半径
    int n_drones_per_group) { // 输入：每组无人机数量

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    //B 是 “每个 j 要处理多少元素”
    //idx 是“一维连续编号”
    //idx / B 就会告诉你现在是第几个 j
    //假设 T=150（时间步），B=64（batch size）
    //总共需要 150 × 64 = 9600 个线程
    //线程 0: j=0, b=0  → 处理时刻 0 的无人机 0
    //线程 1: j=0, b=1  → 处理时刻 0 的无人机 1
    //线程 63: j=0, b=63  → 处理时刻 0 的无人机 63
    //线程 64: j=1, b=0   → 处理时刻 1 的无人机 0
    //线程 9599: j=149, b=63  → 处理时刻 149 的无人机 63
    const int j = idx / B;//时间步索引
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;//当前无人机索引
    // assert(j < pos.size(0));
    // assert(b < pos.size(1));

    const scalar_t ox = pos[j][b][0];//x坐标
    const scalar_t oy = pos[j][b][1];//y坐标
    const scalar_t oz = pos[j][b][2];//z坐标

    scalar_t min_dist = max(1e-3f, oz + 1);// 到地面的距离
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);// 最近点 z（地面）

    // others
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;// 跳过自己
        // 其他无人机的位置
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        // 计算距离（椭球体模型）
        dist = max(1e-3f, sqrt(dist) - r);
        // 更新最近点
        // 从当前障碍物中心指向无人机的向量 direction = (cx - ox, cy - oy, cz - oz)
        if (dist < min_dist) {
            min_dist = dist;
            // 最近点 = 当前位置 + 距离 × 单位方向
            // nearest_pt = origin + dist × direction_normalized
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz;
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy;
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // voxels
    //这段代码计算无人机到AABB（轴对齐包围盒）立方体的最近点，是一个经典的点到长方体距离算法。
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];// 中心
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);// 半边长
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        // 3. 计算距离
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        // 4. 更新全局最近点
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}

//这个 CUDA 内核计算深度图的梯度，用于反向传播。它是实现可微分渲染的关键部分。
//dddp: [B, 3, H, W]  // 深度梯度（例如 [64, 3, 12, 16]）
// 3 个通道：
// dddp[b][0] = ∂depth/∂x（深度对 x 方向的梯度）
// dddp[b][1] = ∂depth/∂y（深度对 y 方向的梯度）
// dddp[b][2]   = ∂depth/∂z（深度对 z 方向的梯度）
template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    const scalar_t unit = fov_x_half_tan / W;
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    // if ReRender.diff_kernel is None:
    //     unit = 0.637 / depth.size(3)
    //     ReRender.diff_kernel = torch.tensor([
    //         [[1, -1], [1, -1]],
    //         [[1, 1], [-1, -1]],
    //         [[unit, unit], [unit, unit]],
    //     ], device=device).mul(0.5)[:, None]
    // ddepthdyz = F.conv2d(depth, ReRender.diff_kernel, None, 2)
    // depth = ddepthdyz[:, 2:]
    // ddepthdyz = torch.cat([
    //     torch.full_like(depth, -1.),
    //     ddepthdyz[:, :2] / depth,
    // ], 1)
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;
    dddp[b][1][u][v] = dddy / dddp_norm;
    dddp[b][2][u][v] = dddz / dddp_norm;
    // ddepthdyz /= ddepthdyz.norm(2, 1, True).clamp_min(8);
}

} // namespace

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
    float fov_x_half_tan) {
    const int threads = 1024;
    size_t state_size = canvas.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(canvas.scalar_type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
    }));
}

void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.scalar_type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,// C++中的PyTorch张量类型
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group) {
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1);
    const dim3 blocks((state_size + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
}
