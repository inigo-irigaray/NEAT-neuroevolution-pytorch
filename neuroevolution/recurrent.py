"""Implementation based PyTorch-NEAT's PyTorch-NEAT/pytorch-neat/recurrent_net.py:
https://github.com/uber-research/PyTorch-NEAT/blob/master/pytorch_neat/recurrent_net.py"""

import torch

from neuroevolution.activation_functions import sigmoid_act
from neuroevolution.graphs import required_for_output
from neuroevolution.utils import make_dense




class RecurrentNetwork:
    def __init__(self, n_in, n_hid, n_out, in2hid, hid2hid, out2hid, in2out, hid2out, out2out, hid_response,
                 out_response, hid_bias, out_bias, batch_size=1, use_current_activs=False, activation=sigmoid_act,
                 n_internal_steps=1, dtype=torch.float64):

        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out

        if n_hid > 0:
            self.in2hid = make_dense((n_hid, n_in), in2hid, dtype=dtype)
            self.hid2hid = make_dense((n_hid, n_hid), hid2hid, dtype=dtype)
            self.out2hid = make_dense((n_hid, n_out), out2hid, dtype=dtype)
            self.hid2out = make_dense((n_out, n_hid), hid2out, dtype=dtype)

        self.in2out = make_dense((n_out, n_in), in2out, dtype=dtype)
        self.out2out = make_dense((n_out, n_out), out2out, dtype=dtype)

        if n_hid > 0:
            self.hid_response = torch.tensor(hid_response, dtype=dtype)
            self.hid_bias = torch.tensor(hid_bias, dtype=dtype)

        self.out_response = torch.tensor(out_response, dtype=dtype)
        self.out_bias = torch.tensor(out_bias, dtype=dtype)

        self.reset(batch_size)

    def reset(self, batch_size=1):
        if self.n_hid > 0:
            self.activs = torch.zeros(batch_size, self.n_hid, dtype=self.dtype)
        else:
            self.activs = None
        self.outputs = torch.zeros(batch_size, self.n_out, dtype=self.dtype)

    def activate(self, inputs):
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=self.dtype)
            activs_for_output = self.activs

            if self.n_hid > 0:
                for _ in range(self.n_internal_steps):
                    self.activs = self.activation(self.hid_response * (
                        self.in2hid.mm(inputs.t()).t() +
                        self.hid2hid.mm(self.activs.t()).t() +
                        self.out2hid.mm(self.outputs.t()).t()) +
                        self.hid_bias)
                if self.use_current_activs:
                    activs_for_output = self.activs

            output_inputs = (self.in2out.mm(inputs.t()).t() +
                             self.out2out.mm(self.outputs.t()).t())

            if self.n_hid > 0:
                output_inputs += self.hid2out.mm(activs_for_output.t()).t()

            self.outputs = self.activation(self.out_response * output_inputs + self.out_bias)

        return self.outputs

    @staticmethod
    def create(genome, config, batch_size=1, activation=sigmoid_act, prune_empty=False,
               use_current_activs=False, n_internal_steps=1):
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)
        if prune_empty:
            nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))

        input_keys = list(genome_config.input_keys)
        hidden_keys = [k for k in genome.nodes.keys() if k not in genome_config.output_keys]
        output_keys = list(genome_config.output_keys)

        hid_response = [genome.nodes[k].response for k in hidden_keys]
        out_response = [genome.nodes[k].response for k in output_keys]

        hid_bias = [genome.nodes[k].bias for k in hidden_keys]
        out_bias = [genome.nodes[k].bias for k in output_keys]

        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    out_bias[i] = 0.0

        n_in = len(input_keys)
        n_hid = len(hidden_keys)
        n_out = len(output_keys)

        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}

        def key_to_idx(key):
            if key in input_keys:
                return input_key_to_idx[key]
            if key in hidden_keys:
                return hidden_key_to_idx[key]
            if key in output_keys:
                return output_key_to_idx[key]

        in2hid = ([], [])
        hid2hid = ([], [])
        out2hid = ([], [])
        in2out = ([], [])
        hid2out = ([], [])
        out2out = ([], [])

        for conn in genome.connections.values():
            if not conn.enabled:
                continue

            i_key, o_key = conn.key
            if o_key not in required and i_key not in required:
                continue
            if prune_empty and i_key not in nonempty:
                print('Pruned {}'.format(conn.key))
                continue

            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)

            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = in2hid
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hid2hid
            elif i_key in output_keys and o_key in hidden_keys:
                idxs, vals = out2hid
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = in2out
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hid2out
            elif i_key in output_keys and o_key in output_keys:
                idxs, vals = out2out
            else:
                raise ValueError('Invalid connection from key {} to key {}'.format(i_key, o_key))

            idxs.append((o_idx, i_idx))
            vals.append(conn.weight)

        return RecurrentNetwork(n_in, n_hid, n_out, in2hid, hid2hid, out2hid, in2out, hid2out, out2out,
                                hid_response, out_response, hid_bias, out_bias, batch_size=batch_size,
                                activation=activation, use_current_activs=use_current_activs,
                                n_internal_steps=n_internal_steps)
