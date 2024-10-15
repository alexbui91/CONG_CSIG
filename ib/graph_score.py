import sys
sys.path.append('..')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

from ib.sim import PAIRWISE_SIMILARITY_FUNCTION

class MLPScore(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPScore, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, embeddings, positive):
        cat_embeddings = torch.cat((embeddings, positive), dim = -1)
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        return pre

class GMNScore(torch.nn.Module):
    def __init__(self, node_state_dim, node_hidden_sizes, sim_func='cosine'):
        super(GMNScore, self).__init__()
        self.sim_func = PAIRWISE_SIMILARITY_FUNCTION[sim_func]
        layers = []
        self._node_state_dim = node_state_dim
        self._node_hidden_sizes = node_hidden_sizes
        layers.append(nn.Linear(self._node_state_dim * 2, self._node_hidden_sizes[0]))
        for i in range(1, len(self._node_hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
        self.mlp = nn.Sequential(*layers)
        self.pooling = dglnn.glob.SumPooling()

    def reset_parameters(self):
        for l in self.mlp.modules():
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

    def compute_cross_attention(self, x, y):
        a = self.sim_func(x, y)
        a_x = torch.softmax(a, dim=1)  # i->j
        a_y = torch.softmax(a, dim=0)  # j->i
        attention_x = torch.mm(a_x, y)
        attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
        return attention_x, attention_y

    def compute_node_update(self, old_context, new_context):
        context = torch.cat([old_context, new_context], dim=1)
        context = self.mlp(context)
        return context

    def forward(self, g1, emb1, g2, emb2):
        attention_x, attention_y = self.compute_cross_attention(emb1, emb2)
        new_context_x, new_context_y = emb1 - attention_x, emb2 - attention_y
        old_context = torch.cat([emb1, emb2], dim=0)
        new_context = torch.cat([new_context_x, new_context_y], dim=0)
        updated_context = self.compute_node_update(old_context, new_context)
        g = dgl.batch([g1, g2], ndata=None, edata=None)
        with g.local_scope():
            graph_emb = self.pooling(g, updated_context)
        x, y = graph_emb[0,:].unsqueeze(dim=1), graph_emb[1,:].unsqueeze(dim=1)
        sim_score = self.sim_func(x.t(), y.t())
        sim_score = sim_score.squeeze(0)
        return sim_score

def log_sinkhorn_norm(log_alpha: torch.Tensor, n_iter: int=20) -> torch.Tensor:
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def gumbel_sinkhorn(log_alpha: torch.Tensor, tau: float = 1.0, n_iter: int = 5, noise: bool = True) -> torch.Tensor:
    sampled_perm_mat = log_sinkhorn_norm(log_alpha, n_iter)
    return sampled_perm_mat

def message_func(edges):
    return {'edge_emb': torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)}