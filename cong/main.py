import sys, time
import argparse
sys.path.append('..')
import torch
from ib.gib_gnns import GIBGCN, GIBSAGE, GIBGIN, GIBGAT, GIBGINE
from ib.utils import load_data, k_fold, load_folds

from cong.utils import cross_validation
from cong.ckpt_node_g import node_g_checkpoints

def list_to_str(lst):
    return ','.join(["%.3f" % i for i in lst])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dataset")
    parser.add_argument("--strategy", default="knn")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--fix_fold", action="store_true")
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--model", type=str, default="GIBGIN")
    parser.add_argument("--gau_scale", type=float, default=8.)
    parser.add_argument("--alpha", type=int, default=3)
    parser.add_argument("--d_metric", type=str, default='gaussian')
    parser.add_argument("--uni", action="store_true", help="Use uniform weights")

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:%i' % args.device)
    batch_size = 128
    dataset_name = args.dataset
    onehot, feat_norm = False, False
    if dataset_name == 'PROTEINS_full':
        feat_norm = True
    elif dataset_name != 'MUTAG' and dataset_name != "Twitter":
        onehot = True

    sample_strategy = args.strategy
    rerank = args.rerank
    edge_assign = args.edge
    fix_fold = args.fix_fold
    fold = -1  # -1 is all
    C, K = args.c, args.k # num centroids, num_neighbors
    dataset, num_classes, dim_feats = load_data(dataset_name, onehot, feat_norm)
    model_map = {
        'GIBGCN': GIBGCN, 
        'GIBSAGE': GIBSAGE, 
        'GIBGIN': GIBGIN,
        'GIBGAT': GIBGAT,
        'GIBGINE': GIBGINE
    }
    md = model_map[args.model]
    ckpt_infor = node_g_checkpoints[dataset_name]
    evals = []
    md_name = md.__qualname__.split('.')[-1]
    if fix_fold:
        if not 'fold' in ckpt_infor[md_name]: raise ValueError("No fold information")
        train, test, val = load_folds(ckpt_infor[md_name]['fold'])
    else:
        train, test, val = k_fold(dataset, 10)
    print("execution model:", md_name)
    val_m, val_std, test_m, test_std = cross_validation(md, ckpt_infor[md_name], dataset, (train, test, val), batch_size, C, K,
                                                        sample_strategy, fold, fix_fold, device, rerank, edge_assign, args.d_metric,
                                                        args.gau_scale, args.alpha, args.uni)
    print("val result",val_m, val_std)
    print("test result", test_m, test_std)
    evals.append("%s val mean: %s, val std: %s, test mean: %s, test std: %s" % (md_name, list_to_str(val_m), list_to_str(val_std), list_to_str(test_m), list_to_str(test_std)))

    tx = int(time.time())
    with open("logs/results_%s_%i.txt" % (dataset_name, tx), 'w') as f:
        f.write(str(args) + "\n" + "\n".join(evals))