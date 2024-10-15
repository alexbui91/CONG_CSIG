import sys
sys.path.append("..")
import time
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler
from torch import tensor
import dgl
from dgl.dataloading import GraphDataLoader
from ib.utils import save_best, mi_donsker, eval_acc, eval_loss

random.seed(1234)

def save(model, args, num_layers, hidden, timestamp, fold, epoch, device):
    mdname = 'IB_' + args.dataset + "_" + str(model) \
                           + "_fold" + str(fold) \
                           + "_" + args.gdist \
                           + '_nlayer' + str(num_layers) \
                           + "_nhidden" + str(hidden) \
                           + '_bs' + str(args.batch_size) \
                           + '_lr' + str(args.lr) \
                           + '_pp' + str(args.pp_weight) \
                           + '_mi' + str(args.mi_weight) + timestamp
    save_best('./ckpt/', model, mdname, fold, epoch, device)

def cross_validation_with_val_set(dataset, model, discriminator, fold_indices, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, inner_loop, mi_weight, pp_weight, size_term=1.,
                                  g_const=False, logger=None, device='', store_model=False, timestamp=0,
                                  num_layers=2, hidden=32, args=None):

    val_losses, accs, durations, train_times, test_times = [], [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*fold_indices)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        val_loader = GraphDataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
        test_loader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=batch_size)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        discriminator.to(device).reset_parameters()
        optimizer_local = Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        cls_loss = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        best_val_loss = float('inf')
        train_time, test_time = 0, 0
        for epoch in range(1, epochs + 1):
            t = time.time()
            train_loss, train_acc = train(model, discriminator, optimizer, optimizer_local, cls_loss, \
                               train_loader, mi_weight, pp_weight, size_term, g_const, inner_loop, device)
            train_time += time.time() - t

            if train_loss != train_loss: # train_loss is nan
                print('NaN')
                continue
            val_loss = eval_loss(model, val_loader, cls_loss, device)
            val_losses.append(val_loss)
            t = time.time()
            test_acc = eval_acc(model, test_loader, device)
            test_time += time.time() - t
            accs.append(test_acc)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'test_acc': test_acc,
            }

            print(eval_info)

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            
            if val_loss < best_val_loss and store_model:
                best_val_loss = val_loss
                save(model, args, num_layers, hidden, timestamp, fold, epoch, device)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        train_times.append(train_time)
        test_times.append(test_time)
        print("train_time", train_time, "test_time", test_time)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    train_times, test_times = tensor(train_times), tensor(test_times)
    n_fold = len(fold_indices[0])
    loss, acc = loss.view(n_fold, epochs), acc.view(n_fold, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(n_fold, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    train_mean, train_std = train_times.mean().item(), train_times.std().item()
    test_mean, test_std = test_times.mean().item(), test_times.std().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f}±{:.3f}, Duration: {:.3f} Train: {:.3f}s±{:.3f}  Test: {:.3f}s±{:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean, train_mean, train_std, test_mean, test_std))

    return loss_mean, acc_mean, acc_std

def train(model, discriminator, optimizer, local_optimizer, cls_loss, loader, mi_weight, pp_weight, size_term, g_const, inner_loop, device=''):
    model.train()
    discriminator.train()

    total_loss = 0
    acc = 0.
    for g, labels in loader:
        g = g.to(device)
        labels = labels.to(device)
        output = model(g, g.ndata['attr'])
        pos_emb = output['sg_embed'] 
        graph_emb = output['g_embed']

        for _ in range(inner_loop):
            local_optimizer.zero_grad()
            mi = mi_donsker(discriminator, graph_emb.detach(), pos_emb.detach(), g.batch_size)
            local_loss = -mi
            local_loss.backward()
            local_optimizer.step()
        optimizer.zero_grad()
        loss = cls_loss(output['sg_pred'], labels) # I(Y, G_s)
        acc += (torch.argmax(output['sg_pred'], dim=1) == labels).to(torch.float32).mean().item()
        if g_const: # add I(Y,G) constraint
            loss += cls_loss(output['g_pred'], labels)
        
        mi_loss = mi_donsker(discriminator, graph_emb, pos_emb, g.batch_size)
        loss = (1-pp_weight) * (loss + mi_weight*mi_loss) + pp_weight * output['pos_penalty'] 
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return total_loss / len(loader), acc / len(loader)