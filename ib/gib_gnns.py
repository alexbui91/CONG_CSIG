import sys
sys.path.append('..')
import torch
from torch import nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GraphConv, GINConv, GATConv, SAGEConv
from dgl.utils import expand_as_pair
from dgl import function as fn

from ib.gib import GIBAbstract
from ib.utils import reset

class GIBGCN(GIBAbstract):
    def __init__(self, dim_nfeats, num_classes, num_layers, hidden, pooling='sum',
                 assignment="softmax", assignment_type="node", temp=1.):
        super(GIBGCN, self).__init__(dim_nfeats, num_classes, num_layers, hidden, pooling, assignment, assignment_type, temp)
        self.conv1 = GraphConv(dim_nfeats, hidden)
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden))

class GIBGIN(GIBAbstract):
    def __init__(self, num_features, num_classes, num_layers, hidden, pooling='sum', assignment="softmax", assignment_type="node", temp=1.):
        super(GIBGIN, self).__init__(num_features, num_classes, num_layers, hidden, pooling, assignment, assignment_type, temp)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), learn_eps=False)
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), learn_eps=False))
    
    def reset_parameters(self):
        reset(self.conv1.apply_func)
        for conv in self.convs:
            reset(conv.apply_func)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

def message_func(edges):
    return {'edge_emb': torch.cat([edges.src['h'], edges.dst['emb']], dim=1)}

class GINEConv(nn.Module):
    def __init__(self, apply_func, apply_edge_func, init_eps=0, learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        self.apply_edge_func = apply_edge_func
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))
        # self.lstm = torch.nn.GRUCell(hidden*2, hidden)

    def reset_parameters(self):
        reset(self.conv1.apply_func)
        for conv in self.convs:
            reset(conv.apply_func)
            reset(conv.apply_edge_func)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        # self.lstm.reset_parameters()

    def message(self, edges):
        r"""User-defined Message Function"""
        edge_emb = torch.cat([edges.src['hn'], edges.dst['hn']], dim=1)
        edge_emb = self.apply_edge_func(edge_emb)
        return {"m": edge_emb}

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, node_feat):
        with graph.local_scope():
            graph.ndata['hn'] = node_feat
            _, feat_dst = expand_as_pair(node_feat, graph)
            graph.update_all(self.message, fn.sum("m", "neigh"))        #sum
            # rst = torch.cat([feat_dst, graph.dstdata["neigh"]])       #cat
            # graph.update_all(self.message, self._lstm_reducer)        #gru
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            rst = self.apply_func(rst)
            return rst

class GIBGINE(GIBGIN):
    def __init__(self, num_features, num_classes, num_layers, hidden, pooling='sum', assignment="softmax", assignment_type="node", temp=1.):
        super(GIBGINE, self).__init__(num_features, num_classes, num_layers, hidden, pooling, assignment, assignment_type, temp)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), learn_eps=False)
        for i in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ),
                    Sequential(
                        Linear(hidden*2, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), learn_eps=False))
            
    def forward(self, g, x):
        x = self.conv1(g, x)
        return self.intermediate(g, x)

class GIBGAT(GIBAbstract):
    def __init__(self, num_features, num_classes, num_layers, hidden, pooling='sum', 
                 assignment="softmax", assignment_type="node", temp=1., num_heads=8, operator='elu'):
        super(GIBGAT, self).__init__(num_features, num_classes, num_layers, hidden, pooling, assignment, assignment_type, temp)
        if operator == 'elu':
            ops = F.elu
        else:
            ops = F.relu
        self.conv1 = GATConv(num_features, hidden, num_heads, 0.3, 0.3, activation=ops) # before relu
        self.conv2 = GATConv(num_heads*hidden, hidden, 1, 0.3, 0.3, activation=ops) # before relu

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, g, x):
        x = self.conv1(g, x)
        s = x.size()
        x = x.view(s[0], s[2]*s[1])
        x = self.conv2(g, x)
        x = x.squeeze()
        return self.intermediate(g, x)
    

class GIBSAGE(GIBAbstract):
    def __init__(self, num_features, num_classes, num_layers, hidden, pooling='sum',
                 assignment="softmax", assignment_type="node", temp=1., aggregator="mean"):
        super(GIBSAGE, self).__init__(num_features, num_classes, num_layers, hidden, pooling, assignment, assignment_type, temp)
        print("Sage Aggregator", aggregator)
        self.conv1 = SAGEConv(num_features, hidden, aggregator)
        self.conv2 = SAGEConv(hidden, hidden, aggregator)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()