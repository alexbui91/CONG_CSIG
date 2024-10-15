import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
import ot

def build_class_index(train_embs, train_labels, num_classes=2, num_centroids=3, train_graphs=None):
    indices, embs, ref_graphs = [], [], []
    for c in range(num_classes):
        label_c_idx = (train_labels == c).nonzero().flatten()
        if len(label_c_idx):
            emb_c = train_embs[label_c_idx]
            if not train_graphs is None:
                ref_graphs.append(train_graphs[label_c_idx])
            else:
                ref_graphs.append(None)
            embs.append(emb_c)
            index = build_index(emb_c, num_centroids, use_gpu=False)
            indices.append(index)
        else:
            indices.append(None)
            embs.append(None)
            ref_graphs.append(None)
    return indices, embs, ref_graphs

def build_index(embeddings, num_centroids=100, use_gpu=False, device=0):
    if not use_gpu and type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
        dim = embeddings[0].shape[0]
    else:
        dim = embeddings.size()[1]
    if use_gpu:
        res = faiss.StandardGpuResources()
        index_flat = faiss.GpuIndexIVFFlat(res, dim, num_centroids, faiss.METRIC_L2)
        # index_flat = faiss.index_cpu_to_gpu(res, device, index_flat)  
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index_flat = faiss.IndexIVFFlat(quantizer, dim, num_centroids, faiss.METRIC_L2) # 100 centroids
    index_flat.train(embeddings)
    index_flat.add(embeddings)
    return index_flat

def search(query, index, k=10):
    if type(query) != np.array:
        query = np.asarray(query, dtype=np.float32)
    _, I = index.search(query, k)
    return I

def augment_queries(xq): 
    extra_column = np.ones((len(xq), 1), dtype=xq.dtype)
    return np.hstack((xq, extra_column))

def augment_database(xb): 
    norms2 = (xb ** 2).sum(1)
    return np.hstack((-2 * xb, norms2[:, None]))

# search farest neighbors
def build_max_index(embeddings, num_centroids=10):
    if type(embeddings) == torch.Tensor:
        embeddings = embeddings.numpy()
        dim = embeddings[0].shape[0]
    quantizer = faiss.IndexFlatIP(dim + 1)
    index = faiss.IndexIVFFlat(quantizer, dim + 1, num_centroids)
    embeddings = augment_database(embeddings)
    index.train(embeddings)
    index.add(embeddings)
    return index    

# search farest neighbors 
# https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34
def search_farest(query, index, k=10):
    query = augment_queries(-query)
    _, I = index.search(query, k)
    # correct the distances since by re-adding the query norm
    # norms2_xq = (xq ** 2).sum(1)
    # Dnew += norms2_xq[:, None]
    return I

def vector_length_normalize(x):
    return torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.FloatTensor([1e-12]).to(x.device))))

def eulidean_distance(x, y):
    s = x - y
    return (s * s).sum(-1)

def eulidean_distance2(x, y):
    s = x - y
    return torch.sqrt((s * s).sum(-1))

def cosine_distance(x, y):
    s = cosine_similarity(x, y)
    return 1. - s

def cosine_similarity(x, y):
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), torch.FloatTensor([1e-12]).to(x.device))))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), torch.FloatTensor([1e-12]).to(x.device))))
    return (x * y).sum(-1)

def sim_cosine(d):
    return d

def sim_minus(d):
    return 1 - d

def sim_inverse(d, **kargs):
    return 1 / (1 + d)

def sim_log(d):
    return torch.log(d + 1) / torch.log(d + 1e-4)

def sim_gausian(d, gau_scale=8):
    return torch.exp(-d / gau_scale) 

def knn(queries, index, k=10):
    refs_g_idx = []
    for i in range(len(queries)):
        ref_g_idx = search(queries[i,:].unsqueeze(0).numpy(), index, k=k)
        ref_g_idx = ref_g_idx[0]
        refs_g_idx.append(ref_g_idx)
    refs_g_tensor = torch.tensor(np.array(refs_g_idx))
    return refs_g_tensor

def extract_centroids(indices):
    centroids = []
    for index in indices:
        c_centroids = get_centroids(index)
        centroids.append(torch.mean(c_centroids, dim=0).unsqueeze(0))
    return centroids

def get_centroids(index):
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    return centroids

def sinkhorn(K, u, v, num_iter=100):
    """
    Compute linear assignment transport between two distributions
    K: M x N
    u: M x 1
    v: N x 1
    T: optimal transport
    """
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(num_iter):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.t(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T 

def weight_normalize(weight):
    return weight / weight.sum()

def emd_similarity(query, context, query_weight, context_weight, distance_metric='cosine', gau_scale=8):
    n, _ = query.size()
    m, d = context.size()
    query = query.tile((1, m)).view(n, m, d)
    if distance_metric == 'cosine':
        sim = (query * context).sum(-1)
        dis = 1. - sim
    else:
        dis = eulidean_distance(query, context)
        sim = sim_gausian(dis, gau_scale)
        # dis = 1 - sim

    # K = torch.exp(-dis / dis.max() / 0.1) 
    # K = torch.exp(-dis / 0.1) 
    # T = sinkhorn(K, query_weight, context_weight)
    T = ot.sinkhorn(query_weight, context_weight, dis, 0.05, numItermax=100, stopThr=1e-1)
    sim = (T * sim).sum().item()
    return sim
    