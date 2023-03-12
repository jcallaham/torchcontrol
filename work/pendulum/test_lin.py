import numpy as np

import torch
from torchdyn.numerics.odeint import odeint
from torchcontrol.systems.classic_control import Pendulum
from torchcontrol.cost import IntegralCost
from torchcontrol.controllers import *

device = torch.device('cpu') # override

# Declaring the cost function
x_star = torch.Tensor([torch.pi, 0]).to(device)
u_star = 0.
cost = IntegralCost(x_star=x_star, u_star=u_star, P=0, Q=1, R=0)

u = NoController().to(device)
pendulum = Pendulum(u, solver='dopri5')

# Stable equilibrium
x0 = torch.Tensor([[0.0, 0.0]])

A, B = pendulum.linearize(x0)
A, B = A.detach().numpy(), B.detach().numpy()
print(A)
print(B)

print(np.linalg.eigvals(A))

# Unstable equilibrium
x0 = torch.Tensor([[torch.pi, 0.0]])

A, B = pendulum.linearize(x0)
A, B = A.detach().numpy(), B.detach().numpy()
print(A)
print(B)

print(np.linalg.eigvals(A))
