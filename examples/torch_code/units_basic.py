import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
m = 5.0
a = 2.0
F = (m * a)
print(F, unit_str='kg·m·s⁻²')
F_bad_typed = (m * a)
print(F_bad_typed, unit_str='m·s⁻²')
v = 3.0
d = 9.0
t = (d / v)
print(t, unit_str='s')
print(d, unit_str='m')
print(v, unit_str='m·s⁻¹')
temp = 100.0
print(((((F * v) * t) * F) * temp), unit_str='m³·kg²·K·s⁻⁴')
F = 3.0
print(F)