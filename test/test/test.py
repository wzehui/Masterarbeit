import torch
import torch.optim as optim
from torch import nn
# print(torch.__version__)

torch.manual_seed(1)    # reproducible
x_c = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=0)  # x data (tensor), shape=(100, 1)
x_u = 5 * x_c + 2 + 0.2*torch.rand(x_c.size())  # noisy y data (tensor), shape=(100, 1)


def model(x, w, b):
    return w * x + b


def loss_fn(x_p, x_c):
    squared_diffs = (x_p - x_c)**2
    return squared_diffs.mean()


params = torch.tensor([1.0, 0.0], requires_grad=True)

nepochs = 100
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
