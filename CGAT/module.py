import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class CGAT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CGAT, self).__init__()
        self.gat = GATConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
