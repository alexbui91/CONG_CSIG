import sys
sys.path.append('..')
import torch
import torch.nn as nn


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
      similarity: a (x, y) -> sim function.

    Raises:
      ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.

    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
      data: NxD float tensor.
      block_idx: N-dim int tensor.
      n_blocks: integer.
      similarity: a string, the similarity metric.

    Returns:
      attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
      ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results

def graph_prop_once(node_states,
                    from_idx,
                    to_idx,
                    message_net,
                    aggregation_module=None,
                    edge_features=None):
    """One round of propagation (message passing) in a graph.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
        row for each node.
      from_idx: [n_edges] int tensor, index of the from nodes.
      to_idx: [n_edges] int tensor, index of the to nodes.
      message_net: a network that maps concatenated edge inputs to message
        vectors.
      aggregation_module: a module that aggregates messages on edges to aggregated
        messages for each node.  Should be a callable and can be called like the
        following,
        `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
        where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
        of the to nodes, i.e. where each message should go to, and n_nodes is an
        int which is the number of nodes to aggregate into.
      edge_features: if provided, should be a [n_edges, edge_feature_dim] float
        tensor, extra features for each edge.

    Returns:
      aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
        aggregated messages, one row for each node.
    """
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    from segment import unsorted_segment_sum
    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor

class GraphPropLayer(nn.Module):
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,  # int
                 node_hidden_sizes,  # int
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net'):
        """Constructor.

        Args:
          node_state_dim: int, dimensionality of node states.
          edge_hidden_sizes: list of ints, hidden sizes for the edge message
            net, the last element in the list is the size of the message vectors.
          node_hidden_sizes: list of ints, hidden sizes for the node update
            net.
          edge_net_init_scale: initialization scale for the edge networks.  This
            is typically set to a small value such that the gradient does not blow
            up.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.
          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphPropLayer, self).__init__()

        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes[:]

        # output size is node_state_dim
        self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type

        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different

        self._layer_norm = layer_norm
        self._prop_type = prop_type
        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._node_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self._node_state_dim * 2, self._node_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = torch.nn.GRU(self._node_state_dim * 3, self._node_state_dim)
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(nn.Linear(self._node_state_dim * 3, self._node_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._node_state_dim * 4, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages(
            self, node_states, from_idx, to_idx, edge_features=None):
        """Compute aggregated messages for each node.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """

        aggregated_messages = graph_prop_once(
            node_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            edge_features=edge_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once(
                node_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                edge_features=edge_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_node_update(self,
                             node_states,
                             node_state_inputs,
                             node_features=None):
        """Compute node updates.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, the input node
            states.
          node_state_inputs: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.

        Raises:
          ValueError: if node update type is not supported.
        """
        if self._node_update_type in ('mlp', 'residual'):
            node_state_inputs.append(node_states)
        if node_features is not None:
            node_state_inputs.append(node_features)

        if len(node_state_inputs) == 1:
            node_state_inputs = node_state_inputs[0]
        else:
            node_state_inputs = torch.cat(node_state_inputs, dim=-1)

        if self._node_update_type == 'gru':
            node_state_inputs = torch.unsqueeze(node_state_inputs, 0)
            node_states = torch.unsqueeze(node_states, 0)
            _, new_node_states = self.GRU(node_state_inputs, node_states)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(node_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_states + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                edge_features=None,
                node_features=None):
        """Run one propagation step.

        Args:
          node_states: [n_nodes, input_node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        return self._compute_node_update(node_states,
                                         [aggregated_messages],
                                         node_features=node_features)

class GraphPropMatchingLayer(GraphPropLayer):
    """A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def forward(self,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                similarity='dotproduct',
                edge_features=None,
                node_features=None):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """
        aggregated_messages = self._compute_aggregated_messages(
            node_states, from_idx, to_idx, edge_features=edge_features)

        cross_graph_attention = batch_block_pair_attention(
            node_states, graph_idx, n_graphs, similarity=similarity)
        attention_input = node_states - cross_graph_attention

        return self._compute_node_update(node_states,
                                         [aggregated_messages, attention_input],
                                         node_features=node_features)

class GraphEmbeddingNet(nn.Module):
    """A graph to embedding mapping network."""

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 prop_type='embedding',
                 name='graph-embedding-net'):
        """Constructor.

        Args:
          encoder: GraphEncoder, encoder that maps features to embeddings.
          aggregator: GraphAggregator, aggregator that produces graph
            representations.

          node_state_dim: dimensionality of node states.
          edge_hidden_sizes: sizes of the hidden layers of the edge message nets.
          node_hidden_sizes: sizes of the hidden layers of the node update nets.

          n_prop_layers: number of graph propagation layers.

          share_prop_params: set to True to share propagation parameters across all
            graph propagation layers, False not to.
          edge_net_init_scale: scale of initialization for the edge message nets.
          node_update_type: type of node updates, one of {mlp, gru, residual}.
          use_reverse_direction: set to True to also propagate messages in the
            reverse direction.
          reverse_dir_param_different: set to True to have the messages computed
            using a different set of parameters than for the forward direction.

          layer_norm: set to True to use layer normalization in a few places.
          name: name of this module.
        """
        super(GraphEmbeddingNet, self).__init__()

        

    
        # name='graph-prop-%d' % layer_id)

    


class GraphMatchingNet(nn.Module):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 node_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 similarity='dotproduct',
                 prop_type='embedding'):
        super(GraphMatchingNet, self).__init__()
        self._encoder = encoder
        self._aggregator = aggregator
        self._node_state_dim = node_state_dim
        self._edge_state_dim = edge_state_dim
        self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        self._n_prop_layers = n_prop_layers
        self._share_prop_params = share_prop_params
        self._edge_net_init_scale = edge_net_init_scale
        self._node_update_type = node_update_type
        self._use_reverse_direction = use_reverse_direction
        self._reverse_dir_param_different = reverse_dir_param_different
        self._layer_norm = layer_norm
        self._prop_layers = []
        self._prop_layers = nn.ModuleList()
        self._layer_class = layer_class
        self._prop_type = prop_type
        self.build_model()
        self._similarity = similarity
    
    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            node_update_type=self._node_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type)
    
    def build_model(self):
        if len(self._prop_layers) < self._n_prop_layers:
            # build the layers
            for i in range(self._n_prop_layers):
                if i == 0 or not self._share_prop_params:
                    layer = self._build_layer(i)
                else:
                    layer = self._prop_layers[0]
                self._prop_layers.append(layer)

    def _apply_layer(self,
                     layer,
                     node_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     edge_features):
        """Apply one layer on the given inputs."""
        return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                     similarity=self._similarity, edge_features=edge_features)

    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs):
        """Compute graph representations.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: [n_edges, edge_feat_dim] float tensor.
          from_idx: [n_edges] int tensor, index of the from node for each edge.
          to_idx: [n_edges] int tensor, index of the to node for each edge.
          graph_idx: [n_nodes] int tensor, graph id for each node.
          n_graphs: int, number of graphs in the batch.

        Returns:
          graph_representations: [n_graphs, graph_representation_dim] float tensor,
            graph representations.
        """

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features

        layer_outputs = [node_states]

        for layer in self._prop_layers:
            # node_features could be wired in here as well, leaving it out for now as
            # it is already in the inputs
            node_states = self._apply_layer(
                layer,
                node_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                edge_features)
            layer_outputs.append(node_states)

        # these tensors may be used e.g. for visualization
        self._layer_outputs = layer_outputs
        return self._aggregator(node_states, graph_idx, n_graphs)

    def reset_n_prop_layers(self, n_prop_layers):
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')
