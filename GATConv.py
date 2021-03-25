import torch
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr='add')#, **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)


    def message(self, edge_index_i, x_i, x_j, size_i):
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        # alpha = F.tanh(alpha)
        # self.alpha = F.leaky_relu(self.alpha)
        # alpha = torch.sigmoid(alpha)
        self.alpha = softmax(self.alpha, edge_index_i, size_i)
        # Sample attention coefficients stochastically.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j*self.alpha.view(-1,1)
        # return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out


