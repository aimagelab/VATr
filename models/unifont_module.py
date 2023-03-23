import torch
import pickle
import numpy as np
from PIL import Image

class UnifontModule(torch.nn.Module):
    def __init__(self, out_dim, alphabet, device='cuda', input_type='unifont', linear=True):
        super(UnifontModule, self).__init__()
        self.device = device
        self.alphabet = alphabet
        self.symbols = self.get_symbols('unifont')
        self.symbols_repr = self.get_symbols(input_type)

        if linear:
            self.linear = torch.nn.Linear(self.symbols_repr.shape[1], out_dim)
        else:
            self.linear = torch.nn.Identity()

    def get_symbols(self, input_type):
        with open(f"files/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32).flatten() for sym in symbols}
        # self.special_symbols = [self.symbols[ord(char)] for char in special_alphabet]
        symbols = [symbols[ord(char)] for char in self.alphabet]
        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().to(self.device)

    def forward(self, QR):
        return self.linear(self.symbols_repr[QR])


class LearnableModule(torch.nn.Module):
    def __init__(self, out_dim, device='cuda'):
        super(LearnableModule, self).__init__()
        self.device = device
        self.param = torch.nn.Parameter(torch.zeros(1, 1, 256, device=device))
        self.linear = torch.nn.Linear(256, out_dim)

    def forward(self, QR):
        return self.linear(self.param).repeat((QR.shape[0], 1, 1))

