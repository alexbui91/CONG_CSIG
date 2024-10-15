node_g_checkpoints = {
    "MUTAG" : {
        "GIBGCN": {
            'path': '../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBGCN_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.11694744062_best.pth',
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/MUTAG_GIBGCN_l_2_h_32_1694744062.pkl'
        },
        "GIBSAGE": {
            'path': "../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBSAGE_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.11694746319_best.pth",
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/MUTAG_GIBSAGE_l_2_h_32_1694746319.pkl'
        },
        "GIBGIN": {
            'path': "../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBGIN_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.051693530511_best.pth",
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/MUTAG_GIBGIN_l_2_h_32_1693530511.pkl'
        },
        "GIBGAT": {
            'path': "../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.11694746547_best.pth",
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1., 8, 'relu'),
            'fold': '../ib/folds/MUTAG_GIBGAT_l_2_h_32_1694746547.pkl'
        },
        "GIBGINE": {
            'path': "../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBGINE_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.051693533852_best.pth",
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': "../ib/folds/MUTAG_GIBGINE_l_2_h_32_1693533852.pkl"
        },
        "GTModel": {
            'path': "../ib/ckpt/mutag_node_g_fold/IB_MUTAG_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs32_lr0.01_pp0.1_mi0.11694746547_best.pth",
            'args': (7, 2, 2, 32, 'sum', "softmax", "node", 1., 8, 'relu'),
            'fold': '../ib/folds/MUTAG_GIBGAT_l_2_h_32_1694746547.pkl'
        },
    },
    "PROTEINS_full": {
        "GIBGCN": {
            'path': '../ib/ckpt/proteins_node_g_fold/IB_PROTEINS_full_GIBGCN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694743878_best.pth',
            'args': (29, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/PROTEINS_full_GIBGCN_l_2_h_32_1694743878.pkl'
        },
        "GIBSAGE": {
            'path': '../ib/ckpt/proteins_node_g_fold/IB_PROTEINS_full_GIBSAGE_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694753949_best.pth',
            'args': (29, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/PROTEINS_full_GIBSAGE_l_2_h_32_1694753949.pkl'
        },
        "GIBGIN": {
            'path': '../ib/ckpt/proteins_node_g_fold/IB_PROTEINS_full_GIBGIN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11693547298_best.pth',
            'args': (29, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/PROTEINS_full_GIBGIN_l_2_h_32_1693547298.pkl'
        },
        "GIBGAT": {
            'path': '../ib/ckpt/proteins_node_g_fold/IB_PROTEINS_full_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694753985_best.pth',
            'args': (29, 2, 2, 32, 'sum', "softmax", "node", 1., 8, 'relu'),
            'fold': '../ib/folds/PROTEINS_full_GIBGAT_l_2_h_32_1694753985.pkl'
        },
    },
    "IMDB-BINARY": {
        "GIBGCN": {
            'path': '../ib/ckpt/imdb_node_g_fold/IB_IMDB-BINARY_GIBGCN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11693547663_best.pth',
            'args': (271, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/IMDB-BINARY_GIBGCN_l_2_h_32_1693547663.pkl'
        },
        "GIBSAGE": {
            'path': '../ib/ckpt/imdb_node_g_fold/IB_IMDB-BINARY_GIBSAGE_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694740484_best.pth',
            'args': (271, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/IMDB-BINARY_GIBSAGE_l_2_h_32_1694740484.pkl'
        },
        "GIBGIN": {
            'path': '../ib/ckpt/imdb_node_g_fold/IB_IMDB-BINARY_GIBGIN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11693554626_best.pth',
            'args': (271, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/IMDB-BINARY_GIBGIN_l_2_h_32_1693554626.pkl'

        },
        "GIBGAT": {
            'path': '../ib/ckpt/imdb_node_g_fold/IB_IMDB-BINARY_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694744805_best.pth',
            'args': (271, 2, 2, 32, 'sum', "softmax", "node", 1., 8, 'relu'),
            'fold': '../ib/folds/IMDB-BINARY_GIBGAT_l_2_h_32_1694744805.pkl'
            
        },
    },
    "DD": {
        "GIBGCN": {
            'path': '../ib/ckpt/dd_node_g_fold/IB_DD_GIBGCN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694739876_best.pth',
            'args': (89, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/DD_GIBGCN_l_2_h_32_1694739876.pkl'
        },
        "GIBSAGE": {
            'path': '../ib/ckpt/dd_node_g_fold/IB_DD_GIBSAGE_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11693555246_best.pth',
            'args': (89, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/DD_GIBSAGE_l_2_h_32_1693555246.pkl'
        },
        "GIBGIN": {
            'path': '../ib/ckpt/dd_node_g_fold/IB_DD_GIBGIN_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694740402_best.pth',
            'args': (89, 2, 2, 32, 'sum', "softmax", "node", 1.),
            'fold': '../ib/folds/DD_GIBGIN_l_2_h_32_1694740402.pkl'
        },
        "GIBGAT": {
            'path': '../ib/ckpt/dd_node_g_fold/IB_DD_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs128_lr0.01_pp0.1_mi0.11694744825_best.pth',
            'args': (89, 2, 2, 32, 'sum', "softmax", "node", 1., 8, 'relu'),
            'fold': '../ib/folds/DD_GIBGAT_l_2_h_32_1694744825.pkl'
        },
    },
    "Twitter": {
        "GIBGCN": {
            'path': "../ib/ckpt/twitter_node_g/IB_Twitter_GIBGCN_fold$$$$_donsker_nlayer2_nhidden128_bs512_lr0.01_pp0.1_mi0.11686878735_best.pth",
            'args': (768, 3, 2, 128, 'max', "softmax", "node", 1.),
            'fold': "../ib/folds/Twitter_GIBGAT_l_2_h_128_1687216907.pkl"
        },
        "GIBSAGE": {
            'path': "../ib/ckpt/twitter_node_g_fold/IB_Twitter_GIBSAGE_fold$$$$_donsker_nlayer2_nhidden2_bs512_lr0.01_pp0.1_mi0.11687216820_best.pth",
            'args': (768, 3, 2, 128, 'max', "softmax", "node", 1., 'gcn'),
            'fold': "../ib/folds/Twitter_GIBSAGE_l_2_h_128_1687216820.pkl"
        },
        "GIBGIN": {
            'path': "../ib/ckpt/twitter_node_g_fold/IB_Twitter_GIBGIN_fold$$$$_donsker_nlayer2_nhidden2_bs512_lr0.01_pp0.1_mi0.11687216761_best.pth",
            'args': (768, 3, 2, 128, 'max', "softmax", "node", 1.),
            'fold': "../ib/folds/Twitter_GIBGIN_l_2_h_128_1687216761.pkl"
        },
        "GIBGAT": {
            'path': "../ib/ckpt/twitter_node_g_fold/IB_Twitter_GIBGAT_fold$$$$_donsker_nlayer2_nhidden2_bs512_lr0.01_pp0.1_mi0.11687216907_best.pth",
            'args': (768, 3, 2, 128, 'max', "softmax", "node", 1., 8, 'relu'),
            'fold': "../ib/folds/Twitter_GIBGAT_l_2_h_128_1687216907.pkl"
        },
    }
}