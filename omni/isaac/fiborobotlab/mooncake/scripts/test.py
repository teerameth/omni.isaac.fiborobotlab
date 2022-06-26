# import omni.isaac.core.utils.torch.rotations as torch_rot
import torch
import torch.nn.functional as f

def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)
def quat_unit(a):
    return normalize(a)
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 1:], p=2, dim=-1), max=1.0))
def q2falling(q):
    # q = f.normalize(q, p=1, dim=1)
    norm_vec = f.normalize(q[:, 1:], p=1, dim=1)
    print(norm_vec)
    return 2 * torch.acos(q[:, 0]) * torch.sqrt((norm_vec[:, 0]*norm_vec[:, 0]+norm_vec[:, 1]*norm_vec[:, 1]))

    # return 2*torch.asin(torch.norm(torch.mul(robots_orientation[:, 1:], up_vectors[:, 1:])))
    # return quat_diff_rad(robots_orientation, up_vectors)

test_a = torch.zeros((1, 4))
test_a[:, 0] = 1
test_b = torch.zeros_like(test_a)
test_b[:, 0] = 0.71
test_b[:, 3] = 0.71
# print(quat_diff_rad(test_a, test_b))
print(q2falling(test_b)/3.14*180)

test_b = torch.zeros_like(test_a)
test_b[:, 0] = 0.71
test_b[:, 2] = 0.71
print(q2falling(test_b)/3.14*180)

test_b = torch.zeros_like(test_a)
test_b[:, 0] = 0.71
test_b[:, 1] = 0.71
print(q2falling(test_b)/3.14*180)

test_b = torch.zeros_like(test_a)
test_b[:, 0] = 0.64
test_b[:, 3] = 0.77
print(q2falling(test_b)/3.14*180)