"""Implementation based on PyTorch-NEAT's PyTorch-NEAT/pytorch-neat/adaptive_linear_net.py:
https://github.com/uber-research/PyTorch-NEAT/blob/master/pytorch_neat/adaptive_linear_net.py"""

import torch

from activation_functions import identity_act, tanh_act
from cppn import clamp_weights_, create_cppn, get_coord_inputs




class AdaptiveLinearNetwork:
    def __init__(self, delta_w_node, in_coords, out_coords, weights_threshold=0.2, weights_max=3.0,
                activation=tanh_act, cppn_activation=identity_act, batch_size=1, device='cuda'):
        self.delta_w_node = delta_w_node
        self.n_in = len(in_coords)
        self.in_coords = torch.tensor(in_coords, dtype=torch.float32, device=device)
        self.n_out = len(out_coords)
        self.out_coords = torch.tensor(out_coords, dtype=torch.float32, device=device)

        self.weights_threshold = weights_threshold
        self.weights_max = weights_max
        self.activation = activation
        self.cppn_activation = cppn_activation

        self.batch_size = batch_size
        self.device = device

        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

        n_in = len(in_coords)
        n_out = len(out_coords)

        zeros = torch.zeros((n_out, n_in), dtype=torch.float32, device=self.device)

        weights = self.cppn_activation(w_node(x_out=x_out, y_out=y_out, x_in=x_in, y_in=y_in,
                                              pre=zeros, post=zeros, w=zeros))

        clamp_weights_(weights, self.weights_threshold, self.weights_max)

        return weights

    def reset(self):
        with torch.no_grad():
            self.in2out = (self.get_init_weights(
                          self.in_coords, self.out_coords, self.delta_w_node).unsqueeze(0).expand(
                          self.batch_size, self.n_out, self.n_in)
                          )

            self.w_expressed = self.in2out != 0
            self.batched_coords = get_coord_inputs(self.in_coords, self.out_coords, self.batch_size)

    def activate(self, inputs):
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32,
                                 device=self.device).unsqueeze(2)

            outputs = self.activation(self.in2out.matmul(inputs))

            in_activs = inputs.transpose(1, 2).expand(self.batch_size, self.n_out, self.n_in)
            out_activs = outputs.expand(self.batch_size, self.n_out, self.n_in)

            (x_out, y_out), (x_in, y_in) = self.batched_coords

            delta_w = self.cppn_activation(self.delta_w_node(x_out=x_out, y_out=y_out,
                                          x_in=x_in, y_in=y_in, pre=in_activs, post=out_activs,
                                          w=self.in2out)
                                          )

            self.delta_w = delta_w

            self.in2out[self.w_expressed] += delta_w[self.w_expressed]
            clamp_weights_(self.in2out, weights_threshold=0.0, weights_max=self.weights_max)

        return outputs.squeeze(2)

    @staticmethod
    def create(config, genome, in_coords, out_coords, weights_threshold=0.2, weights_max=3.0,
               out_activation=None, activation=tanh_act, cppn_activation=identity_act,
               batch_size=1, device='cuda'):
        nodes = create_cppn(config, genome, ['x_in', 'y_in', 'x_out', 'y_out', 'pre', 'post', 'w'],
                            ['delta_w'], out_activation=out_activation)
        delta_w_node = nodes[0]

        return AdaptiveLinearNetwork(delta_w_node, in_coords, out_coords,
                                    weights_threshold=weights_threshold,
                                    weights_max=weights_max, activation=activation,
                                    cppn_activation=cppn_activation, batch_size=batch_size,
                                    device=device)
