import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp

from ib.gib import GIBAbstract


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def reset_parameters(self):
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def reset_parameters(self):
        self.MHA.reset_parameters()
        self.FFN1.reset_parameters()
        self.FFN2.reset_parameters()

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)

class GTModel(GIBAbstract):
    def __init__(
        self,
        dim_nfeats,
        num_classes,
        num_layers=8,
        hidden=32,
        pooling="sum",
        assignment_type="softmax",
        aggregate_type="node",
        temp=1.,
        pos_enc_size=8,
        num_heads=8,
    ):
        super().__init__(dim_nfeats, num_classes, num_layers, hidden, pooling, assignment_type, aggregate_type, temp)
        self.input_linear = nn.Linear(dim_nfeats, hidden)
        self.pos_enc_size = pos_enc_size
        self.pos_linear = nn.Linear(pos_enc_size, hidden)
        self.conv1 = GTLayer(hidden, num_heads)
        for _ in range(num_layers-1):
            self.convs.append(GTLayer(hidden, num_heads))

        self.build_pred_layers(hidden, num_classes)

    def build_pred_layers(self, hidden, num_classes):
        if self.aggregate_type == 'node':
            input_size = hidden
        else: # edge selection
            input_size = hidden * 2
        self.cluster1 = nn.Linear(input_size, hidden)
        if self.assignment_type == 'softmax':
            self.cluster2 = nn.Linear(hidden, 2)
        else:
            self.cluster2 = nn.Linear(hidden, 1)
        self.lin1 = nn.Linear(hidden, hidden//2)
        self.lin2 = nn.Linear(hidden//2, num_classes)
        self.mse_loss = nn.MSELoss()

    def forward(self, g, attr):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        node_enc = self.input_linear(attr)
        pos_enc = dgl.lap_pe(g, k=self.pos_enc_size, padding=True).to(g.device)
        pos_enc = self.pos_linear(pos_enc)
        h = node_enc + pos_enc
        h = self.conv1(A, h)
        for layer in self.convs:
            h = layer(A, h)
        return self.intermediate(g, h)