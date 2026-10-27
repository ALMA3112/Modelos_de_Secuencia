import torch

class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.l(x)
