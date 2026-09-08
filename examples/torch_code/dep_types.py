import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def dot(u, v, n=None):
    if n is None:
        n = int(u.shape[0])
    return torch.sum(torch.stack([torch.as_tensor((u[int(i)] * v[int(i)])) for i in range(int(n))]).float())

def append(u, v, n=None):
    if n is None:
        n = int(u.shape[0])
    return torch.cat([u, torch.stack([torch.as_tensor(v)])])

def f(n, v, m=None):
    if m is None:
        m = int(v.shape[0])
    return m

def outer3(u, v, w, m=None, n=None, p=None):
    if m is None:
        m = int(u.shape[0])
    if n is None:
        n = int(v.shape[0])
    if p is None:
        p = int(w.shape[0])
    return torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(torch.stack([torch.as_tensor(((u[int(i)] * v[int(j)]) * w[int(k)])) for k in range(int(p))]).float()) for j in range(int(n))])) for i in range(int(m))])

def append_row(x, row, m=None, n=None):
    if m is None:
        m = int(x.shape[0])
    if n is None:
        n = int(x.shape[1])
    return torch.cat([x, torch.stack([torch.as_tensor(row)])])

# === Program ===
v3 = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
w3 = torch.tensor([4.0, 5.0, 6.0], device=DEVICE)
e = 3.0
print(dot(v3, w3, 3))
print(append(v3, e, 3))
print(append(append(v3, e, 3), e, 4))
a = 5.0
print(f(a, v3, 3))
print(outer3(v3, torch.stack([torch.as_tensor(1.0), torch.as_tensor(0.0)]), torch.stack([torch.as_tensor(1.0), torch.as_tensor(2.0), torch.as_tensor(3.0)]), 3, (1 + 1), (2 + 1)))
x2mat = torch.tensor([[3.0, 2.0], [1.0, 2.0], [1.0, 2.0]], device=DEVICE)
new_row = torch.tensor([1.0, 5.0], device=DEVICE)
print(append_row(x2mat, new_row, 3, 2))