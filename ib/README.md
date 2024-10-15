# IMPLEMENTATION OF INFORMATION BOTTLENECK USING DGL

***Using --g_const to add I(Y^,G) constraint***

**1. Node assignment**
```
python main.py --dataset Twitter --batch_size 512 --device 1 --model sage --pp_weight 0.1 --mi_weight 0.1 --inner_loop 20 --hidden_size 128 --pooling max --sage_aggregator gcn --g_const
```

**2. Edge assignment**

```
python main.py --dataset MUTAG --batch_size 32 --device 1 --model gcn --pp_weight 0.005 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh
python main.py --dataset MUTAG --batch_size 32 --device 1 --model gcn --pp_weight 0.005 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh --g_const
```

```
python main.py --dataset PROTEINS_full --batch_size 128 --device 0 --model gcn --pp_weight 0.001 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh -feat_norm
python main.py --dataset PROTEINS_full --batch_size 128 --device 0 --model gcn --pp_weight 0.001 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh -feat_norm --g_const
python main.py --dataset PROTEINS_full --batch_size 128 --device 2 --model gat --pp_weight 0.001 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh --feat_norm --num_heads 2 --gat_ops relu
python main.py --dataset PROTEINS_full --batch_size 128 --device 2 --model gat --pp_weight 0.001 --inner_loop 20 --size_term 0. --assignment_type edge --assignment tanh --feat_norm --num_heads 2 --gat_ops relu --g_const
```