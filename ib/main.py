import sys, time
sys.path.append('..')
from itertools import product

import argparse
import torch

from ib.train_eval import cross_validation_with_val_set

from ib.graph_score import MLPScore
from ib.gib_gnns  import GIBGAT, GIBGCN, GIBGIN, GIBSAGE, GIBGINE
from ib.graph_transformer import GTModel

from ib.utils import load_data, k_fold, save_folds

parser = argparse.ArgumentParser()
# training
parser.add_argument('--device', type=int, default=0) 
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--epochs', type=int, default=100) 
parser.add_argument('--inner_loop', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128) 
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--store_model', action='store_true')
# feat
parser.add_argument('--onehot', action='store_true', help='Turn feature into onehot (IMDB)')
parser.add_argument('--feat_norm', action='store_true', help='Std norm features')
# mask
parser.add_argument('--assignment', type=str, default='softmax', help='softmax|sigmoid')
parser.add_argument('--assignment_type', type=str, default='node', help='Select nodes or select edges node|edge')
parser.add_argument('--temp', type=float, default=1., help="temperature for gumbel softmax")
# model
parser.add_argument('--model', type=str, default='gcn', help="gcn,sage,gin,gat,gine,all")
parser.add_argument('--hidden_size', type=int, default=32) 
parser.add_argument('--pooling', type=str, default='sum', help='sum|max|mean')
# gat
parser.add_argument('--num_heads', type=int, default=8, help='Num heads of GAT')
parser.add_argument('--gat_ops', type=str, default='relu', help='Operator for GAT [elu, relu]')
# sage 
parser.add_argument('--sage_aggregator', type=str, default='mean', help='mean|gcn|pool|lstm')
# graph transformer
parser.add_argument('--pe', type=int, default=2)
# regularization
parser.add_argument('--mi_weight', type=float, default=0.1)
parser.add_argument('--pp_weight', type=float, default=0.3)
parser.add_argument('--gdist', type=str, default='donsker', help='Graph distance operator donsker|gmn|isonet')
parser.add_argument('--size_term', type=float, default=1., help='Control the size loss by a factor')
parser.add_argument('--g_const', action='store_true', help='Using original graph constraint')
parser.add_argument('--weight_decay', type=float, default=0., help="using for L2 regularization") # setting to 0.001 improve training perf by reducing overfitting

args = parser.parse_args()
print(args)

device = torch.device('cuda:%i' % args.device if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

layers = [2]
hiddens = [args.hidden_size]
datasets = [args.dataset]
if args.model == 'all':
    nets = [GIBGCN, GIBSAGE, GIBGIN, GIBGAT]
else:
    models = { 'gcn': GIBGCN, 'sage': GIBSAGE, 'gin': GIBGIN, 'gat': GIBGAT, 'gine': GIBGINE, 'gt': GTModel}
    nets = [models[args.model]]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))
timestamp = str(int(time.time()))
results = []
kargs = {}
if args.model == 'gat':
    kargs['num_heads'] = args.num_heads
    kargs['operator'] = args.gat_ops
elif args.model == 'sage':
    kargs['aggregator'] = args.sage_aggregator
elif args.model == 'gt':
    kargs['num_heads'] = args.num_heads
    kargs['pos_enc_size'] = args.pe
    
for dataset_name, Net in product(datasets, nets):

    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))

    dataset, num_classes, dim_nfeats = load_data(args.dataset, args.onehot, args.feat_norm)
    print("Num graphs %i num feats %i" % (len(dataset), dim_nfeats))
    for num_layers, hidden in product(layers, hiddens):
        model = Net(dim_nfeats, num_classes, num_layers, hidden, args.pooling,
                    args.assignment, args.assignment_type, args.temp, **kargs)
        input_mlp = hidden * 2
        discriminator = MLPScore(input_mlp, hidden)
        fold_indices = k_fold(dataset, 10)
        if args.store_model:
            save_folds(fold_indices, dataset_name, str(model), num_layers, hidden, timestamp)
        
        loss, acc, std = cross_validation_with_val_set(
            dataset,
            model,
            discriminator,
            fold_indices=fold_indices,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=args.weight_decay,
            inner_loop = args.inner_loop,
            mi_weight = args.mi_weight,
            pp_weight=args.pp_weight,
            size_term=args.size_term,
            g_const=args.g_const,
            logger= None,
            device=device,
            store_model=args.store_model,
            timestamp=timestamp,
            num_layers=num_layers,
            hidden=num_layers,
            args=args
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)

    desc = '{:.3f} , {:.3f}'.format(best_result[1], best_result[2])
    print('Best result - {}'.format(desc))
    results += ['{} - {}: {}'.format(dataset_name, model, desc)]
print('-----\n{}'.format('\n'.join(results)))

if len(nets) == 1:
    suffx = args.dataset + "_" + str(model) + "_" + args.gdist
else:
    suffx = args.dataset + "_" + args.gdist
log_file_name = 'IB_' + suffx \
                      + '_nlayer' + str(num_layers) \
                      + "_nhidden" + str(hidden) \
                      + '_bs' + str(args.batch_size) \
                      + '_lr' + str(args.lr) + "_" \
                      + '_pp' + str(args.pp_weight) \
                      + '_mi' + str(args.mi_weight) + timestamp
with open("./logs/%s.txt" % log_file_name, 'w') as f:
    f.write(str(args) + '\n' + '\n'.join(results))
