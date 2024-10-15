import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, ReLU, BatchNorm1d as BN
import dgl.nn.pytorch as dglnn
import dgl.function as fn

from ib.utils import concrete

class GIBAbstract(torch.nn.Module):
    def __init__(self, dim_nfeats, num_classes, num_layers, hidden, pooling, assignment_type='softmax', aggregate_type='node', temp=1.):
        super(GIBAbstract, self).__init__()
        self.conv1 = None
        self.convs = nn.ModuleList()
        self.assignment_type = assignment_type
        self.temp = temp
        self.aggregate_type = aggregate_type # default is similar to GIB
        self.pooling_style = pooling
        self.pooling_ops = {
            'sum': dglnn.glob.SumPooling(),
            'mean': dglnn.glob.AvgPooling(),
            'max': dglnn.glob.MaxPooling()
        }
        self.build_pred_layers(hidden, num_classes)

    def __repr__(self):
        return self.__class__.__name__
    
    def build_pred_layers(self, hidden, num_classes):
        if self.aggregate_type == 'node':
            input_size = hidden
        else: # edge selection
            input_size = hidden * 2
        self.cluster1 = Linear(input_size, hidden)
        if self.assignment_type == 'softmax':
            self.cluster2 = Linear(hidden, 2)
        else:
            self.cluster2 = Linear(hidden, 1)
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.mse_loss = nn.MSELoss()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.cluster1.reset_parameters()
        self.cluster2.reset_parameters()

    def get_entropy_loss(self, pos, neg):
        pos, neg = pos + 1e-8, neg + 1e-8
        mask_ent = -pos * torch.log(pos) - neg * torch.log(neg)
        return mask_ent.sum()

    def assignment_transform(self, x):
        return self.cluster2(torch.relu(self.cluster1(x)))

    def assignment(self, x):
        output = self.assignment_transform(x)
        if self.training:
            output = concrete(output, self.temp)
        if self.assignment_type == 'softmax':
            output = torch.nn.functional.softmax(output, dim=1)
        elif self.assignment_type == 'tanh':
            output = torch.tanh(output)
        else:
            output = F.sigmoid(output)
            output_neg = 1. - output
            output = torch.cat([output, output_neg], dim=1)
        return output

    def aggregate_node(self, g, x, assignment):
        EYE = torch.ones((g.batch_size, 2)).to(x.device)
        # S^T x A x S
        with g.local_scope():
            g.ndata['prob']  = assignment
            # sta computation
            g.update_all(fn.copy_u('prob', 'm'), fn.sum('m', 'prob_sum'))
            sta = g.ndata['prob_sum'].unsqueeze(dim=2) # Nx2x1
            assignment = assignment.unsqueeze(dim=1) # Nx1x2
            stas = sta @ assignment # Nx2x2
            g.ndata['connectivity'] = stas.reshape(g.num_nodes(), 4)
            new_adj = self.pooling_ops['sum'](g, g.ndata['connectivity'])
            new_adj = new_adj.reshape(g.batch_size, 2, 2)
            # S^T x X
            assignment = torch.permute(assignment, (0, 2, 1))
            graph_pos_features = assignment @ x.unsqueeze(dim=1)
            filter_features = graph_pos_features[:,0,:].squeeze()
            pos_embedding = self.pooling_ops[self.pooling_style](g, filter_features)
            graph_embedding = self.pooling_ops[self.pooling_style](g, x)
        # calculate connectivity loss
        normalize_new_adj = F.normalize(new_adj, p=1, dim=2, eps = 0.00001)
        norm_diag = torch.diagonal(normalize_new_adj, dim1=1, dim2=2)
        pos_penalty = self.mse_loss(norm_diag, EYE)

        return pos_embedding, graph_embedding, pos_penalty
    
    def aggregate_edge(self, g, x):
        msg = lambda edge: {'edge_emb': torch.cat([edge.src['emb'], edge.dst['emb']], dim=1)}
        with g.local_scope():
            g.ndata['emb'] = x
            g.apply_edges(msg)
            # compute edge mask
            edge_assignment = self.assignment(g.edata.pop('edge_emb')) # E x 2
            # compute new edge embedding using mask & old node_embedding
            g.apply_edges(fn.u_add_v('emb', 'emb', 'sum_emb'))
            pos_penalty = 0.
            if self.assignment_type == 'tanh':
                g.edata['agg_emb'] = (edge_assignment * g.edata['sum_emb']) # ExD
            else:
                g.edata['agg_emb'] = (edge_assignment[:,0].unsqueeze(1) * g.edata['sum_emb']) # ExD
            # pass edge embeddings to nodes & sub at nodes
            g.update_all(fn.copy_e('agg_emb', 'agg_emb_t'), fn.sum('agg_emb_t', 'pos_emb'))
            # global embeddings
            pos_embedding = self.pooling_ops[self.pooling_style](g, g.ndata['pos_emb'])
            graph_embedding = self.pooling_ops[self.pooling_style](g, x)
            # size_loss = 0.1 * edge_assignment[:,0].sum()
        return pos_embedding, graph_embedding, pos_penalty

    def predict(self, g_embedding):
        x = F.relu(self.lin1(g_embedding))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def intermediate(self, g, x):
        if self.aggregate_type == 'node':
            assignment = self.assignment(x)
            pos_embedding, graph_embedding, pos_penalty = self.aggregate_node(g, x, assignment)
            # print(assignment)
            # assignment idx 0 => selected; 1 not selected
            selected_nodes = 1. - torch.argmax(assignment, dim=1) 
        else:
            assignment = None
            pos_embedding, graph_embedding, pos_penalty = self.aggregate_edge(g, x)
            selected_nodes = None
        node_emb = x
        s_out = self.predict(pos_embedding) # pred of subgraph embeddings
        g_out = self.predict(graph_embedding) # pred of orig graph embeddings 
        return {
            "sg_pred": s_out,
            "g_pred": g_out,
            "sg_embed": pos_embedding,
            "g_embed": graph_embedding,
            "pos_penalty": pos_penalty,
            "assignment": assignment,
            "n_embed": node_emb,
            "selected_nodes": selected_nodes
        }

    def forward(self, g, x):
        x = self.conv1(g, x)
        for conv in self.convs:
            x = conv(g, x)
        return self.intermediate(g, x)
        