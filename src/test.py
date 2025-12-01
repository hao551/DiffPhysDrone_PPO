import math
import torch
import quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply

R = torch.randn((64, 3, 3), dtype=torch.double, device='cuda')
dg = torch.randn((64, 3), dtype=torch.double, device='cuda')
z_drag_coef = torch.randn((64, 1), dtype=torch.double, device='cuda')
drag_2 = torch.randn((64, 2), dtype=torch.double, device='cuda')
pitch_ctl_delay = torch.randn((64, 1), dtype=torch.double, device='cuda')
g_std = torch.tensor([[0, 0, -9.80665]], dtype=torch.double, device='cuda')
act_pred = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
act = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
p = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
v_wind = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)
a = torch.randn((64, 3), dtype=torch.double, device='cuda', requires_grad=True)

grad_decay = 0.4
ctl_dt = 1/15

def run_forward_pytorch(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt):
    alpha = torch.exp(-pitch_ctl_delay * ctl_dt)
    act_next = act_pred * (1 - alpha) + act * alpha
    # dg = dg * math.sqrt(1 - ctl_dt) + torch.randn_like(dg) * 0.2 * math.sqrt(ctl_dt)
    v_fwd_s, v_left_s, v_up_s = (v.add(-v_wind)[:, None] @ R).unbind(-1)
    # 0.047 = (4*rotor_drag_coefficient*motor_velocity_real) / sqrt(9.8)
    drag = drag_2[:, :1] * (v_fwd_s.abs() * v_fwd_s * R[..., 0] + v_left_s.abs() * v_left_s * R[..., 1] + v_up_s.abs() * v_up_s * R[..., 2] * z_drag_coef)
    drag += drag_2[:, 1:] * (v_fwd_s * R[..., 0] + v_left_s * R[..., 1] + v_up_s * R[..., 2] * z_drag_coef)
    a_next = act_next + dg - drag
    p_next = g_decay(p, grad_decay ** ctl_dt) + v * ctl_dt + 0.5 * a * ctl_dt**2
    v_next = g_decay(v, grad_decay ** ctl_dt) + (a + a_next) / 2 * ctl_dt
    return act_next, p_next, v_next, a_next

act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, 0)

_act_next, _p_next, _v_next, _a_next = run_forward_pytorch(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt)

assert torch.allclose(act_next, _act_next)
assert torch.allclose(a_next, _a_next)
assert torch.allclose(p_next, _p_next)
assert torch.allclose(v_next, _v_next)

d_act_next = torch.randn_like(act_next)
d_p_next = torch.randn_like(p_next)
d_v_next = torch.randn_like(v_next)
d_a_next = torch.randn_like(a_next)

torch.autograd.backward(
    (_act_next, _p_next, _v_next, _a_next),
    (d_act_next, d_p_next, d_v_next, d_a_next),
)

d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
    R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next, grad_decay, ctl_dt)

assert torch.allclose(d_act_pred, act_pred.grad)
assert torch.allclose(d_act, act.grad)
assert torch.allclose(d_p, p.grad)
assert torch.allclose(d_v, v.grad)
assert torch.allclose(d_a, a.grad)
