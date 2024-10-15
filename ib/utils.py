import os, random
import pickle as pkl
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import dgl
from dgl.data import GINDataset, TUDataset
from dig.xgraph.dataset import SentiGraphDataset


class DGLSentiDataset:
    def __init__(self, graphs, labels, num_classes, name='Senti'):
        self._name = name
        self.num_classes = num_classes
        self.graphs = graphs
        self.labels = labels
        self.num_graphs = len(self.graphs)
        self.dim_nfeats = self.graphs[0].ndata['attr'].size()[1]

    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, i):
        return (self.graphs[i], self.labels[i])

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

def concrete(adj, bias=0., beta=1.):
    random_noise = torch.rand(adj.size()).to(adj.device)
    if bias > 0. and bias < 0.5:
        r = 1 - bias - bias
        random_noise = r * random_noise + bias
    gate_inputs = torch.log(random_noise) - torch.log(1 - random_noise)
    gate_inputs = (gate_inputs + adj) / beta
    return gate_inputs

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    n = len(dataset)
    labels = [l for _, l in dataset]
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(n), labels):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))
    return train_indices, test_indices, val_indices

def save_folds(fold_indices, dataset_name, model_name, num_layers, hidden, timestamp):
    file_name = "./folds/%s_%s_l_%i_h_%i_%s.pkl" % (dataset_name, model_name, num_layers, hidden, timestamp)
    with open(file_name, 'wb') as f:
        pkl.dump(fold_indices, f)
    
def load_folds(path):
    with open(path, 'rb') as f:
        train, test, val = pkl.load(f)
    return train, test, val

def save_best(ckpt_dir, model, model_name, fold, epoch, device=None):
    print('saving....')
    model.to('cpu')
    state = {
        'net': model.state_dict(),
        'fold': fold,
        'epoch': epoch
    }
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, best_pth_name)
    torch.save(state, ckpt_path)
    model.to(device)

def load_gin_data(dataset):
    dataset = GINDataset(dataset, True, False)
    graphs = []
    for g in dataset.graphs:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        graphs.append(g)
    num_classes = dataset.num_classes
    dim_nfeats = dataset.dim_nfeats
    dataset.graphs = graphs
    return dataset, num_classes, dim_nfeats

def load_sentiment_data(dataset):
    data = SentiGraphDataset('../data', 'Graph-Twitter')
    graphs, labels = [], []
    for d in data:
        src, dst = d.edge_index[0,:], d.edge_index[1,:]
        g = dgl.graph((src, dst))
        g.ndata['attr'] = d.x.to(torch.float32)
        labels.append(d.y.item())
        graphs.append(g)
    labels = torch.LongTensor(labels) 
    dataset = DGLSentiDataset(graphs, labels, labels.max().item() + 1, 'Graph-Twitter')
    return dataset, dataset.num_classes, dataset.dim_nfeats
    
def load_tu_data(dataset, onehot=False, feat_norm=False):
    dataset = TUDataset(dataset)
    dataset.graph_labels = dataset.graph_labels.flatten().numpy()
    graphs = []
    if onehot: # for dd
        if 'node_labels' in dataset[0][0].ndata:
            node_labels = torch.cat([g.ndata['node_labels'] for g in dataset.graph_lists], dim=0)
        else:
            node_labels = torch.tensor([g.in_degrees(g.nodes()).max() for g in dataset.graph_lists])    
        node_label_max = node_labels.max().item()
    else: # for imdb
        degrees = torch.cat([g.in_degrees(g.nodes()).to(torch.float32) for g in dataset.graph_lists], dim=0)
        d_m, d_std = degrees.mean(), degrees.std()
    
    if feat_norm: # for proteins
        feats = torch.cat([g.ndata['node_attr'] for g in dataset.graph_lists], dim=0)
        d_m, d_std = feats.mean(), feats.std()
    for g in dataset.graph_lists:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        if 'node_attr' in g.ndata:
            feat = g.ndata.pop('node_attr').to(torch.float32)
            if feat_norm:
                feat = (feat - d_m) / d_std
            g.ndata['attr'] = feat
        else:
            if onehot:
                if 'node_labels' in g.ndata:
                    feats = torch.nn.functional.one_hot(g.ndata['node_labels'].squeeze(), node_label_max+1)
                else:
                    feats = torch.nn.functional.one_hot(g.in_degrees(g.nodes()) - 1, node_label_max+1)
                g.ndata['attr'] = feats.to(torch.float32)
            else:
                in_degrees = g.in_degrees(g.nodes()).to(torch.float32)
                g.ndata['attr'] = ((in_degrees - d_m) / d_std).view(-1, 1)
        graphs.append(g)
    num_classes = dataset.num_labels
    dim_nfeats = graphs[0].ndata['attr'].size()[1]
    dataset.graph_lists = graphs
    return dataset, num_classes, dim_nfeats

def load_data(dataset, onehot=False, feat_norm=False):
    args = {}
    if dataset == "MUTAG":
        func = load_gin_data
    elif dataset.lower() == "twitter":
        func = load_sentiment_data
    else:
        func = load_tu_data
        args = {'onehot': onehot, 'feat_norm': feat_norm}
    dataset, num_classes, dim_nfeats = func(dataset, **args)
    return dataset, num_classes, dim_nfeats

def execute_model(dataloader, model, device=None):
    preds, embs, subgraphs, labels = [], [], [], []
    model.eval()
    graphs = []
    for g, lbs in dataloader:
        # step 1: extract prototype & project in an inner loop
        g = g.to(device)
        labels.append(lbs)
        with torch.no_grad():
            outputs = model(g, g.ndata['attr'])
            embs.append(outputs['sg_embed'].detach())
            g.ndata['selected_nodes'] = outputs['selected_nodes'].detach()
            g.ndata['emb'] = outputs['n_embed'].detach()
            pred = torch.argmax(outputs['sg_pred'], dim=1)
            preds.append(pred)
            
        all_graphs = dgl.unbatch(g)
        for g in all_graphs:
            sn = (g.ndata.pop('selected_nodes') == 1).nonzero().flatten().to(torch.int64)
            sg = dgl.node_subgraph(g, sn)
            subgraphs.append(sg)
            graphs.append(g)

    prot_embs = torch.cat(embs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    return graphs, prot_embs, subgraphs, preds, labels

def execute_model2(dataloader, model, device=None):
    preds, embs, subgraphs, labels = [], [], [], []
    model.eval()
    graphs = []
    for g, lbs in dataloader:
        # step 1: extract prototype & project in an inner loop
        g = g.to(device)
        labels.append(lbs)
        with torch.no_grad():
            outputs = model(g, g.ndata['attr'])
            embs.append(outputs['sg_embed'].detach())
            g.ndata['selected_nodes'] = outputs['selected_nodes'].detach()
            g.ndata['emb'] = outputs['n_embed'].detach()
            g.ndata['assignment'] = outputs['assignment'].detach()
            pred = torch.argmax(outputs['sg_pred'], dim=1)
            preds.append(pred)
            
        all_graphs = dgl.unbatch(g)
        for g in all_graphs:
            sn = (g.ndata.pop('selected_nodes') == 1).nonzero().flatten().to(torch.int64)
            sg = dgl.node_subgraph(g, sn)
            subgraphs.append(sg)
            graphs.append(g)

    prot_embs = torch.cat(embs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    return {
        "graphs": np.array(graphs),
        "graph_embs": prot_embs,
        "subgraphs": subgraphs,
        "preds": preds,
        "labels": labels
    }

def load_backbone(model, ckpt, ckpt_args, device=None):
    backbone = model(*ckpt_args)
    checkpoint = torch.load(ckpt)
    print("start loading backbone model %s" % str(backbone))
    backbone.load_state_dict(checkpoint['net'])
    backbone = backbone.to(device) 
    return backbone

def mi_est(joint, margin):
    # v = torch.mean(torch.exp(margin))
    # fix nan issue
    n = margin.size()[0]
    mx = margin.max()
    v1 = torch.mean(joint)
    v2 = mx + torch.log(torch.sum(torch.exp(margin - mx))) - torch.log(torch.tensor([n], dtype=torch.float32).to(margin.device))
    est = v1 - v2
    return est

def mi_donsker(discriminator, embs, positive, num_graphs):
    shuffle_embs = embs[torch.randperm(num_graphs)]
    joint = discriminator(embs, positive)
    margin = discriminator(shuffle_embs, positive)
    return mi_est(joint, margin)

def mi_gmn(discriminator, subgraphs, pos_graphs, detach=False):
    pos_scores, neg_scores = [], []
    l = len(subgraphs)
    idx = list(range(l))
    random.shuffle(idx)
    for i in range(l):
        sg, pg = subgraphs[i], pos_graphs[i]
        ng = pos_graphs[idx[i]]
        sg_emb, pg_emb, ng_emb = sg.ndata['emb'], pg.ndata['emb'], ng.ndata['emb']
        if detach:
            sg_emb, pg_emb, ng_emb = sg_emb.detach(), pg_emb.detach(), ng_emb.detach()
        pos_score = discriminator(sg, sg_emb, pg, pg_emb)
        neg_score = discriminator(sg, sg_emb, ng, ng_emb)
        pos_scores.append(pos_score)
        neg_scores.append(neg_score)
    joint = torch.cat(pos_scores, dim=0)
    margin = torch.cat(neg_scores, dim=0)
    return mi_est(joint, margin)

def eval_acc(model, loader, device=''):
    model.eval()

    correct = 0
    for data in loader:
        g, labels = data
        g = g.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(g, g.ndata['attr'])
            pred = output['sg_pred']
            pred = torch.argmax(pred, 1)
            acc = ((pred == labels).to(torch.float)).mean()
        correct += acc.item()
    correct = correct / len(loader)
    return correct


def eval_loss(model, loader, cls_loss, device=''):
    model.eval()
    loss = 0
    for data in loader:
        g, labels = data
        g = g.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output = model(g, g.ndata['attr'])
            out = output['sg_pred']
        loss += cls_loss(out, labels).item()
    return loss / len(loader.dataset)

def unpack_graphs(g, node_emb, selected_nodes):
    with g.local_scope():
        g.ndata['selected'] = selected_nodes
        g.ndata['emb'] = node_emb
        graphs = dgl.unbatch(g)
        sub_graphs, pos_graphs = [], []
        for i, pg in enumerate(graphs):
            node_idx = (pg.ndata['selected'] == 0).to(torch.int64).nonzero().flatten()
            sg = dgl.node_subgraph(pg, node_idx, output_device=g.device)
            sub_graphs.append(sg)
            pos_graphs.append(pg)
    return sub_graphs, pos_graphs