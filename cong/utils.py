import os, sys, time
sys.path.append('..')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import Sampler

import dgl
from dgl.dataloading import GraphDataLoader

import ot

from ib.utils import execute_model, execute_model2

from cong.similarity import   build_index, build_class_index, knn, eulidean_distance, \
                            cosine_distance, cosine_similarity, sim_inverse, sim_gausian, \
                            sim_minus, sim_log, sim_cosine, get_centroids, weight_normalize, \
                            vector_length_normalize, emd_similarity

sim_metrics = [sim_inverse, sim_gausian] # sim_inverse, sim_gausian, sim_minus, sim_log, sim_cosine

class TestSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.num_samples = len(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

def load_backbone(model_cls, path, args, device=None):
    print(model_cls, path, args)
    checkpoint = torch.load(path)
    backbone = model_cls(*args)
    backbone.load_state_dict(checkpoint['net'])
    if not device is None:
        backbone = backbone.to(device)
    backbone.eval()
    return backbone

def sim_pred(d, ref_labels, true_labels, num_classes, sim_func, **kargs):
    pros_distance_sim = sim_func(d, **kargs)
    pros_distance_topk_att = torch.softmax(pros_distance_sim, dim=1)
    ref_labels = F.one_hot(ref_labels, num_classes).to(torch.float32)
    pros_distance_topk_att = pros_distance_topk_att.unsqueeze(1) # Nx1xR
    preds = torch.argmax((pros_distance_topk_att @ ref_labels).squeeze(1), dim=1)
    pred_acc = (preds == true_labels).to(torch.float32).mean()
    return pred_acc, preds

def meta_pred(query_embs, support_set, ref_labels, true_labels, num_classes, k, **kargs):
    emb_map = torch.tile(query_embs, (1, k)).view(query_embs.size()[0], k, query_embs.size()[-1])
    accs, preds = [], []
    for func in sim_metrics:
        if func != sim_cosine:
            d = eulidean_distance(support_set, emb_map)
        else:
            d = cosine_similarity(support_set, emb_map)
        acc, pred = sim_pred(d, ref_labels, true_labels, num_classes, func, **kargs)
        accs.append(acc.item())
        preds.append(pred)
    return accs, preds

# compute similarity for emd
def one_to_many(vg, ref_graphs, edge_assign=False, d_metric='gassian', gau_scale=4, uni=False):
    if not edge_assign:
        vg_n_emb = vg.ndata['emb']
        vg_n_weight = vg.ndata['assignment'][:,0]
    else:
        vg_n_emb = vg.edata['edge_emb']
        vg_n_weight = vg.edata['edge_assignment'][:,0]
    vg_n_emb = vector_length_normalize(vg_n_emb)
    if uni:
        vg_n_weight = (torch.ones_like(vg_n_weight) / len(vg_n_weight)).to(vg_n_weight.device)
    else:
        vg_n_weight = weight_normalize(vg_n_weight)
    similarities = []
    for i, rg in enumerate(ref_graphs):
        if not edge_assign:
            rg_n_emb = rg.ndata['emb']
            rg_n_weight = rg.ndata['assignment'][:,0]
        else:
            rg_n_emb = rg.edata['edge_emb']
            rg_n_weight = rg.edata['edge_assignment'][:,0]

        rg_n_emb = vector_length_normalize(rg_n_emb)
        if uni:
            rg_n_weight = (torch.ones_like(rg_n_weight) / len(rg_n_weight)).to(rg_n_weight.device)
        else:
            rg_n_weight = weight_normalize(rg_n_weight)
        sim = emd_similarity(vg_n_emb, rg_n_emb, vg_n_weight, rg_n_weight, d_metric, gau_scale)
        similarities.append((sim, i))
    return similarities
    
def select_top_k_sim(similarities, ref_lbs, k=10):
    tmp_lb, tmp_distance = [], []
    for s, i in sorted(similarities, reverse=True)[:k]:
        tmp_lb.append(ref_lbs[i])
        tmp_distance.append(s)
    return tmp_distance, tmp_lb

def knn_pred(embs, true_labels, index, index_embs, index_labels, train_graphs, val_graphs, num_classes=2, k=10, rerank=False,
             edge_assign=False, d_metric='gaussian', gau_scale=4, alpha=3, uni=False):
    if rerank:
        k = k * alpha
    # preprare support set
    refs = knn(embs, index, k=k)
    support_set = index_embs[refs]
    ref_labels = index_labels[refs]

    if not rerank:
        return meta_pred(embs, support_set, ref_labels, true_labels, num_classes, k)
    else:
        ref_graphs = train_graphs[refs]
        filter_labels, distances = [], []
        for vg, refs, ref_lbs in zip(val_graphs, ref_graphs, ref_labels):
            similarities = one_to_many(vg, refs, edge_assign, d_metric, gau_scale, uni)
            tmp_distance, tmp_lb = select_top_k_sim(similarities, ref_lbs, k)
            filter_labels.append(tmp_lb)
            distances.append(tmp_distance)
        device = embs.device
        distances = torch.tensor(distances).to(device)
        filter_labels = torch.tensor(filter_labels, dtype=torch.int64).to(device)
        acc, pred = sim_pred(distances, filter_labels, true_labels, num_classes, sim_cosine)
        return [acc.item()], [pred]
    

def knnc_pred(embs, true_labels, index, index_embs, ref_graphs, val_graphs, num_classes=2, k=10, rerank=False,
              edge_assign=False, d_metric='gaussian', gau_scale=4, alpha=3, uni=False):
    # embs = embs.to('cpu')
    k = k // len(index)
    if not rerank:
        # prepare support sets
        ref_labels, support_set = [], []
        valid_i = 0
        for i, (idx, emb) in enumerate(zip(index, index_embs)):
            if idx is None: continue
            refs = knn(embs, idx, k=k)
            sup_set = emb[refs]
            support_set.append(sup_set)
            ref_labels.append(torch.ones_like(refs) * i)
            valid_i += 1
        support_set = torch.cat(support_set, dim=1)
        ref_labels = torch.cat(ref_labels, dim=1)
        k = k * valid_i
        return meta_pred(embs, support_set, ref_labels, true_labels, num_classes, k)
    else:
        device = embs.device
        # prepare support sets
        ref_labels, distances = [], []
        k1 = k * alpha
        for i, (idx, emb, graphs) in enumerate(zip(index, index_embs, ref_graphs)):
            if idx is None: continue
            refs = knn(embs, idx, k=k1)
            # select highest similar in each cls
            sample_distances = []
            for vg, refs in zip(val_graphs, graphs[refs]):
                similarities = one_to_many(vg, refs, edge_assign, d_metric, gau_scale)
                selected_sim = [s for s, _ in sorted(similarities, reverse=True)[:k]]
                sample_distances.append(selected_sim)
            sample_distances = torch.tensor(sample_distances).to(device)
            distances.append(sample_distances)
            ref_labels.append(torch.ones_like(sample_distances, dtype=torch.int64) * i)

        distances = torch.cat(distances, dim=1)
        ref_labels =  torch.cat(ref_labels, dim=1)
        acc, pred = sim_pred(distances, ref_labels, true_labels, num_classes, sim_cosine)
        return [acc.item()], [pred]

def k_centroid_pred(embs, true_labels, indices, num_classes=2, C=10, gau_scale=4): # C ~ num_centroids
    all_centroids = []
    ref_labels = []
    for i, idx in enumerate(indices):
        centroids = get_centroids(idx)
        all_centroids.append(centroids)
        ref_labels.append(torch.ones((1,), dtype=torch.long) * i)

    support_set = torch.cat(all_centroids, dim=0)
    N = len(embs)
    R = int(C * len(indices))
    D = embs[0].size()[-1]
    support_set = torch.tile(support_set, (N, 1)).view(N, R, D)
    emb_map = torch.tile(embs, (1, R)).view(N, R, D)
    accs, preds = [], []
    ref_labels = torch.tile(torch.cat(ref_labels, dim=0), (N, 1)).view(N, num_classes)
    for func in sim_metrics:
        if func != sim_cosine:
            d = eulidean_distance(support_set, emb_map) # N x R
            d = d.view(N, num_classes, C)
            d = d.min(dim=2).values
        else:
            d = cosine_similarity(support_set, emb_map) # N x R
            d = d.view(N, num_classes, C)
            d = d.max(dim=2).values
        acc, pred = sim_pred(d, ref_labels, true_labels, num_classes, func, gau_scale=gau_scale)
        accs.append(acc.item())
        preds.append(pred)
    return accs, preds

def apply_edges(graphs):
    def message_func(edges):
        return {'edge_emb': torch.cat([edges.src['emb'], edges.dst['emb']], dim=1)}
    
    for g in graphs:
        g.apply_edges(message_func)
        g.apply_edges(dgl.function.u_add_v('assignment', 'assignment', 'edge_assignment'))

def execute_fold(backbone, dataset, train_idx, test_idx, batch_size, C, K, sample_strategy='knn',
                 device=None, rerank=False, edge_assign=False, d_metric='gaussian', gau_scale=4, alpha=3, uni=False):
    train_sampler = TestSampler(train_idx)
    test_sampler = TestSampler(test_idx)
    train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=batch_size)
    train_outputs = execute_model2(train_loader, backbone, device)
    train_graphs, train_prot_embs, train_preds, train_labels = train_outputs['graphs'], train_outputs['graph_embs'], train_outputs['preds'], train_outputs['labels']
    test_outputs = execute_model2(test_loader, backbone, device)
    test_graphs, test_prot_embs, t_preds, test_labels = test_outputs['graphs'], test_outputs['graph_embs'], test_outputs['preds'], test_outputs['labels']
    if edge_assign:
        apply_edges(train_graphs)
        apply_edges(test_graphs)
    t_pred_acc = (t_preds.to('cpu') == test_labels).to(torch.float32).mean().item()

    # create index for trained samples
    train_prot_embs = train_prot_embs.to('cpu')
    test_prot_embs = test_prot_embs.to('cpu')
    if sample_strategy == 'knn':
        index = build_index(train_prot_embs, num_centroids=C, use_gpu=False)
        t_acc, _ = knn_pred(test_prot_embs, test_labels, index, train_prot_embs, train_labels, train_graphs, test_graphs, dataset.num_classes, k=K,
                            rerank=rerank, edge_assign=edge_assign, d_metric=d_metric, gau_scale=gau_scale, alpha=alpha, uni=uni)
    elif sample_strategy == 'knnc':
        train_labels = train_preds.to('cpu')
        indices, ref_embs, ref_graphs = build_class_index(train_prot_embs, train_labels, dataset.num_classes, C, train_graphs)
        # search similar concepts for val
        t_acc, _ = knnc_pred(test_prot_embs, test_labels, indices, ref_embs, ref_graphs, test_graphs, dataset.num_classes, k=K, rerank=rerank,
                             edge_assign=edge_assign, d_metric=d_metric, gau_scale=gau_scale, alpha=alpha, uni=uni)
    else:
        train_labels = train_preds.to('cpu') # using pred_labels for knnc and k centroids is better
        indices, ref_embs, _ = build_class_index(train_prot_embs, train_labels, dataset.num_classes, C)
        t_acc, _ = k_centroid_pred(test_prot_embs, test_labels, indices, dataset.num_classes, C, gau_scale=gau_scale)

    return t_acc, t_pred_acc

def cross_validation(md, ckpt_infor, dataset, fold_info, batch_size=32, C=3, K=10, sample_strategy='knn',
                    fold=-1, fix_fold=False, device=None, rerank=False, edge_assign=False, d_metric='gaussian',
                    gau_scale=4, alpha=3, uni=False):
    ckpt_path = ckpt_infor['path']
    args = ckpt_infor['args']
    train, test, _ = fold_info
    test_res, t_preds = [], [], []
    exe_times = []
    if fix_fold:
        if fold != -1:
            ckpt_path = ckpt_path.replace("$$$$", str(fold))
            backbone = load_backbone(md, ckpt_path, args, device)
            backbone.eval()
            print("fold %i" % fold)
            train_idx, test_idx = train[fold], test[fold]
            s = time.time()
            for i in range(len(train)):
                t_acc, t_pred_acc = execute_fold(backbone, dataset, train_idx, test_idx, batch_size,
                                                        C, K, sample_strategy, device, rerank, edge_assign, d_metric,
                                                        gau_scale, alpha, uni)
                test_res.append(t_acc)
                t_preds.append(t_pred_acc)
            t = time.time() - s
            print("execution time: %.2f" % t)
            
        else:
            for i in range(len(train)):
                tmp_path = ckpt_path.replace("$$$$", str(i))
                args = ckpt_infor['args']
                backbone = load_backbone(md, tmp_path, args, device)
                backbone.eval()
                print("fold %i" % i)
                train_idx, test_idx = train[i], test[i]
                t = time.time()
                t_acc, t_pred_acc = execute_fold(backbone, dataset, train_idx, test_idx, batch_size,
                                                        C, K, sample_strategy, device, rerank, edge_assign, d_metric,
                                                        gau_scale, alpha, uni)
                t = time.time() - t
                print({'test_acc': t_acc, 'model_pred_acc': t_pred_acc, 'execution_time': t})
                exe_times.append(t)
                test_res.append(t_acc)
                t_preds.append(t_pred_acc)
    else:
        backbone = load_backbone(md, ckpt_path, args, device)
        backbone.eval()
        test_res, t_preds = [], [], []
        train, test, _ = fold_info
        for i, (train_idx, test_idx) in enumerate(zip(train, test)):
            print("fold %i" % i)
            t = time.time()
            t_acc, t_pred_acc = execute_fold(backbone, dataset, train_idx, test_idx, batch_size,
                                                    C, K, sample_strategy, device, rerank, edge_assign, d_metric,
                                                    gau_scale, alpha, uni)
            exe_times.append(time.time() - t)
            test_res.append(t_acc)
            t_preds.append(t_pred_acc)
    test_m, test_std = np.mean(test_res, axis=0), np.std(test_res, axis=0)    
    t_pred_m, t_pred_std = np.mean(t_preds), np.std(t_preds)   
    time_mean, time_std = np.mean(exe_times), np.std(exe_times)
    print("acc by ib", t_pred_m, t_pred_std)
    print("Execution time %.3f±%.3f" % (time_mean, time_std))
    return test_m, test_std