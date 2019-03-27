import torch
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
# print(torch.__version__)

torch.manual_seed(1)    # reproducible

pi = 3.14
omega = 1 * pi
phi = 0.5 * pi
A = 3
n = torch.unsqueeze(torch.linspace(1, 10, 100), dim=1)  # x data (tensor), shape=(100, 1)

x_c = A * torch.sin(omega * n + phi)
x_u = A * torch.sin(omega * n + phi) + 0.5 * torch.randn(n.size())
# plt.plot(n.squeeze().numpy(), x_u.squeeze().numpy(), '.')
# plt.show()


def model(x, w, b):
    return w * x + b


def loss_fn(x_p, x_c):
    squared_diffs = (x_p - x_c)**2
    return squared_diffs.mean()


params = torch.tensor([1.0, 1.0], requires_grad=True)

nepochs = 500
learning_rate = 1e-1

optimizer = optim.Adam([params], lr=learning_rate)

for epoch in range(nepochs):
    # forward
    x_p = model(x_u, *params)
    loss = loss_fn(x_p, x_c)

    print('Epoch %d, Loss %f' % (epoch, float(loss)))

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_p = model(x_u, *params)
print(params)
