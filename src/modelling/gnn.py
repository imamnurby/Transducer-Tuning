from .norm import RMSNorm
import torch
from torch_geometric.nn import GCN, GATv2Conv

class GVE_GAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float = 0):
        super().__init__()
        self.layer_norm = RMSNorm(input_dim)
        self.down_proj = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim//n_heads, n_heads, concat=True, bias=False, dropout=dropout)
        self.up_proj = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=False)

    def forward(self, x, edge_index):
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.conv1(x, edge_index)
        x = self.up_proj(x)
        return x

class GVE_Linear(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float = 0):
        super().__init__()
        self.layer_norm = RMSNorm(input_dim)
        self.down_proj = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.conv1 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim//n_heads, bias=False)
        self.up_proj = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=False)

    def forward(self, x, edge_index):
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.conv1(x)
        x = self.up_proj(x)
        return x


class GVE_GCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float = 0):
        super().__init__()
        self.layer_norm = RMSNorm(input_dim)
        self.down_proj = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        self.conv1 = GCN(hidden_dim, hidden_dim, num_layers=1)
        self.up_proj = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=False)

    def forward(self, x, edge_index):
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.conv1(x, edge_index)
        x = self.up_proj(x)
        return x
    
class NormSingleGAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float = 0):
        super().__init__()
        self.layer_norm = RMSNorm(input_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, n_heads, concat=False, bias=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.layer_norm(x)
        x = self.conv1(x, edge_index)
        return x

GNN_MAPPING = {
    "GVE_GCN": GVE_GCN,
    "GVE_GAT": GVE_GAT,
    "GVE_Linear": GVE_Linear
}
