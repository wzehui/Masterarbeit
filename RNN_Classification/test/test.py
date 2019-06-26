import torch
import random

data = torch.ones(400)

index = index = [n for n in range(len(data))]
temp = []

random.seed(5)  # reproducibility

for i in range(10):
    temp += random.choices(index, k=len(index))

index_test = list(set(index)-set(temp))

