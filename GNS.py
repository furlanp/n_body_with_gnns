import torch
import torch.nn as nn
import torch_scatter
import torch_geometric as pyg
from torchvision.ops import MLP

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)

class InteractionNetwork(pyg.nn.MessagePassing):
    def __init__(self, dim):
       super().__init__()
       self.lin_node = MLP(2 * dim, dim, dim)

    def message(self, x_i, x_j):
        x = torch.cat((x_i, x_j), dim=-1)
        x = self.lin_node(x)
        return x

    def aggregate(self, inputs, index):
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce='sum')

    def forward(self, x, edge_index):
        aggr = self.propagate(edge_index, x=(x, x))
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        node_out = x + node_out
        return node_out

class GNS(torch.nn.Module):
   # Graph Network-based Simulators(GNS) 
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_layers):
        super().__init__()
        self.node_in = MLP(input_dim, hidden_dim, hidden_dim)
        self.layers = torch.nn.ModuleList([InteractionNetwork(hidden_dim) for _ in range(gnn_layers)])
        self.node_out = MLP(hidden_dim, hidden_dim, output_dim)
 
    def forward(self, node_feature, edge_index):
        node_feature = self.node_in(node_feature)   # encoder
        for layer in self.layers:   # processor
            node_feature = layer(node_feature, edge_index)
        out = self.node_out(node_feature)   # decoder
        return out