# pip install torch matplotlib  # (matplotlib just for plotting)
import math, torch, random
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- Ground-truth SDFs (analytic) ----
def sdf_sphere(x, r=0.6):
    # x: (...,3)
    return torch.linalg.norm(x, dim=-1) - r


def sdf_box(x, b=(0.4, 0.25, 0.2)):
    # axis-aligned box centered at origin
    q = torch.abs(x) - torch.tensor(b, device=x.device)
    return torch.linalg.norm(torch.clamp(q, min=0), dim=-1) + torch.clamp(
        q.max(dim=-1).values, max=0
    )


def sdf_union(a, b):  # union of two SDFs
    return torch.minimum(a, b)


# ---- Neural SDF f_theta: R^3 -> R ----
class SDFNet(nn.Module):
    def __init__(self, width=256, depth=5):
        super().__init__()
        layers = []
        in_dim = 3
        for i in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.ReLU(inplace=True)]
            in_dim = width
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


net = SDFNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)


def sample_points(n):
    # sample near surface (important in practice)
    xyz = torch.empty(n, 3).uniform_(-1.0, 1.0)
    # bias some points near the surface of sphere for better learning
    dirs = torch.randn(n, 3)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    near = 0.6 * dirs + 0.03 * torch.randn(n, 3)  # near sphere surface
    mask = torch.rand(n) < 0.5
    xyz[mask] = near[mask]
    return xyz


for step in range(5000):
    x = sample_points(8192).to(device)
    gt = sdf_union(
        sdf_sphere(x), sdf_box(x - torch.tensor([0.25, 0, 0], device=device))
    )
    pred = net(x)
    # clamp far distances to focus capacity near surface (as in DeepSDF-style)
    delta = 0.1
    loss = (
        (torch.clamp(pred, -delta, delta) - torch.clamp(gt, -delta, delta)).abs().mean()
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(step, float(loss))

# ---- Visualize a 2D slice (z=0 plane) of SDF contours ----
with torch.no_grad():
    xs = torch.linspace(-1, 1, 200)
    ys = torch.linspace(-1, 1, 200)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    Z = torch.zeros_like(X)
    grid = torch.stack([X, Y, Z], dim=-1).to(device).reshape(-1, 3)
    S = net(grid).reshape(200, 200).cpu()

plt.figure()
plt.contour(
    X.numpy(), Y.numpy(), S.numpy(), levels=[-0.05, 0.0, 0.05]
)  # zero contour is the surface
plt.title("Learned SDF slice at z=0 (0-level is surface)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
