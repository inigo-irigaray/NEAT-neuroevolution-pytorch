"""Implementation based on Uber AI's Lab Pytorch-NEAT pytorch-neat/cppn.py:
https://github.com/uber-research/PyTorch-NEAT/blob/master/pytorch_neat/cppn.py"""

import torch

from neuroevolution.activation_functions import str_to_activation
from neuroevolution.aggregations import str_to_aggregation
from neuroevolution.graphs import required_for_output




class Node:
    def __init__(self, children, weights, response, bias, activation, aggregation, name=None, leaves=None):
        self.children = children
        self.weights = weights
        self.response = response
        self.bias = bias
        self.activation = activation
        self.activation_name = activation
        self.aggregation = aggregation
        self.aggregation_name = aggregation
        self.name = name
        if leaves is not None:
            assert isinstance(leaves, dict)
        self.leaves = leaves
        self.activs = None
        self.is_reset = None

    def __repr__(self):
        header = "Node(%s, response=%d, bias=%d, activation=%s, aggregation=%s)" % (
                 self.name, self.response, self.bias, self.activation_name, self.aggregation_name)
        child_reprs = []
        for w, child in zip(self.weights, self.children):
            child_reprs.append("    <- {} * " % (w) + repr(child).replace("\n", "\n    "))
        return header + "\n" + "\n".join(child_reprs)

    def activate(self, xs, shape):
        if not xs:
            return torch.full(shape, self.bias)
        inputs = [w * x for w, x in zip(self.weights, xs)]
        try:
            pre_activs = self.aggregation(inputs)
            activs = self.activation(self.response * pre_activs + self.bias)
            assert activs.shape == shape, "Wrong shape for node %s" % self.name
        except Exception:
            raise Exception("Failed to activate node %s" % self.name)
        return activs

    def get_activs(self, shape):
        if self.activs is None:
            xs = [child.get_activs(shape) for child in self.children]
            self.activs = self.activate(xs, shape)
        return self.activs

    def __call__(self, **inputs):
        assert self.leaves is not None
        assert inputs
        shape = list(inputs.values())[0].shape
        self.reset()
        for name in self.leaves.keys():
            assert (inputs[name].shape==shape), "Wrong activs shape for leaf %s, %d != %d" % (
            name, inputs[name].shape, shape)
            self.leaves[name].set_activs(inputs[name])
        return self.get_activs(shape)

    def _prereset(self):
        if self.is_reset is None:
            self.is_reset = False
            for child in self.children:
                child._prereset()

    def _postreset(self):
        if self.is_reset is not None:
            self.is_reset = None
            self.activs = None
            for child in self.children:
                child._postreset()

    def _reset(self):
        if not self.is_reset:
            self.is_reset = True
            self.activs = None
            for child in self.children:
                child._reset()

    def reset(self):
        self._prereset()
        self._reset()
        self._postreset()




class Leaf:
    def __init__(self, name=None):
        self.activs = None
        self.name = name

    def __repr__(self):
        return "Leaf(%s)" % self.name

    def set_activs(self, activs):
        self.activs = activs

    def get_activs(self, shape):
        assert self.activs is not None, "Missing activs for leaf node %s" % self.name
        assert (self.activs.shape==shape), "Wrong activs shape for lead node %s, %d != %d" % (
                                           self.name, self.activs.shape, shape)
        return self.activs

    def _prereset(self):
        pass

    def _postreset(self):
        pass

    def _reset(self):
        self.activs = None

    def reset(self):
        self._reset()




def create_cppn(config, genome, leaf_names, node_names, out_activation=None):
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys,
                                   genome.connections)
    # define coordinate frames for the phenotype from inputs and expressed genes in the genome
    node_inputs = {inp: [] for inp in genome_config.output_keys}
    for conngene in genome.connections.values():
        if not conngene.enabled: # skip disabled genes in the genome
            continue

        inp, out = conngene.key # key = tuple(conn_inp_node, conn_out_node)
        if out not in required and inp not in required:
            continue

        if inp in genome_config.output_keys:
            continue

        if out not in node_inputs:
            node_inputs[out] = [(inp, conngene.weight)]
        else: # fill in genome output genes that were created empty beforehand
            node_inputs[out].append((inp, conngene.weight))

        if inp not in node_inputs:
            node_inputs[inp] = []

    nodes = {inp: Leaf() for inp in genome_config.input_keys}

    assert len(leaf_names) == len(genome_config.input_keys)
    leaves = {name: nodes[inp] for name, inp in zip(leaf_names, genome_config.input_keys)}

    def build_node(idx):
        if idx in nodes:
            return nodes[idx]
        node = genome.nodes[idx]
        conns = node_inputs[idx]
        children = [build_node(i) for i, _ in conns]
        weights = [w for _, w in conns]
        if idx in genome_config.output_keys and out_activation is not None:
            activation = out_activation
        else:
            activation = str_to_activation[node.activation]
        aggregation = str_to_aggregation[node.aggregation]
        nodes[idx] = Node(children, weights, node.response, node.bias, activation, aggregation, leaves=leaves)
        return nodes[idx]

    # retroactively build graph
    for idx in genome_config.output_keys:
        build_node(idx)

    outputs = [nodes[i] for i in genome_config.output_keys]

    for name in leaf_names:
        leaves[name].name = name

    for i, name in zip(genome_config.output_keys, node_names):
        nodes[i].name = name

    return outputs




def clamp_weights_(weights, weights_threshold=0.2, weights_max=3.0):
    low_idxs = weights.abs() < weights_threshold
    weights[low_idxs] = 0
    weights[weights > 0] -= weights_threshold
    weights[weights < 0] += weights_threshold
    weights[weights > weights_max] = weights_max
    weights[weights < -weights_max] = -weights_max




def get_coord_inputs(in_coord, out_coord, batch_size=None):
    n_in = len(in_coord)
    n_out = len(out_coord)

    if batch_size is not None:
        in_coord = in_coord.unsqueeze(0).expand(batch_size, n_in, 2)
        out_coord = out_coord.unsqueeze(0).expand(batch_size, n_out, 2)

        x_out = out_coord[:, :, 0].unsqueeze(2).expand(batch_size, n_out, n_in)
        y_out = out_coord[:, :, 1].unsqueeze(2).expand(batch_size, n_out, n_in)
        x_in = in_coord[:, :, 0].unsqueeze(1).expand(batch_size, n_out, n_in)
        y_in = in_coord[:, :, 1].unsqueeze(1).expand(batch_size, n_out, n_in)
    else:
        x_out = out_coord[:, 0].unsqueeze(1).expand(n_out, n_in)
        y_out = out_coord[:, 1].unsqueeze(1).expand(n_out, n_in)
        x_in = in_coord[:, 0].unsqueeze(0).expand(n_out, n_in)
        y_in = in_coord[:, 1].unsqueeze(0).expand(n_out, n_in)

    return (x_out, y_out), (x_in, y_in)
