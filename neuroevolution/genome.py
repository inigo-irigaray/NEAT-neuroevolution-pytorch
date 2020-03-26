"""Implementation based on NEAT-Python neat-python/neat/genome.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/genome.py"""

import random
import sys
from itertools import count

import neuroevolution.activation_functions as activation_functions
import neuroevolution.aggregations as aggregations
import neuroevolution.genes as genes
import neuroevolution.graphs as graphs
from neuroevolution.config import ConfigParameter, write_pretty_params




class DefaultGenomeConfig:
    allowed_connectivity = ["unconnnected", "fs_neat_nohidden", "fs_neat", "fs_neat_hidden", "full_nodirect",
                            "full", "full_direct", "partial_nodirect", "partial", "partial_direct"]

    def __init__(self, params):
        self.activation_defs = activation_functions.str_to_activation
        self.aggregation_defs = aggregations.str_to_aggregation
        self._params = [ConfigParameter("n_in", int),
                        ConfigParameter("n_out", int),
                        ConfigParameter("n_hid", int),
                        ConfigParameter("feed_forward", bool),
                        ConfigParameter("compatibility_disjoint_coefficient", float),
                        ConfigParameter("compatibility_weight_coefficient", float),
                        ConfigParameter("conn_add_prob", float),
                        ConfigParameter("conn_delete_prob", float),
                        ConfigParameter("node_add_prob", float),
                        ConfigParameter("node_delete_prob", float),
                        ConfigParameter("single_structural_mutation", bool, "false"),
                        ConfigParameter("structural_mutation_surer", str, "default"),
                        ConfigParameter("initial_connection", str, "unconnected")]

        self.node_gene_type = params["node_gene_type"]
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params["connection_gene_type"]
        self._params += self.connection_gene_type.get_config_params()

        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.input_keys = [-i - 1 for i in range(self.n_in)]
        self.output_keys = [i for i in range(self.n_out)]

        self.connection_fraction = None

        if "partial" in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError("'partial' connection must between [0.0, 1.0]")

        assert self.initial_connection in self.allowed_connectivity

        if self.structural_mutation_surer.lower() in ["true", "1", "yes", "on"]:
            self.structural_mutation_surer = "true"
        elif self.structural_mutation_surer.lower() in ["false", "0", "no", "off"]:
            self.structural_mutation_surer = "false"
        elif self.structural_mutation_surer.lower() == "default":
            self.structural_mutation_surer = "default"
        else:
            raise RuntimeError("Invalid structural_mutation_surer %r" % self.structural_mutation_surer)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs[name] = func

    def add_aggregation(self, name, func):
        self.aggregation_defs[name] = func

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError("'partial' connection value must be between [0.0, 1.0]")
            f.write("initial_connection = {0} {1} \n".format(self.initial_connection, self.connection_fraction))
        else:
            f.write("initial_connection = {0} \n".format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity
        write_pretty_params(f, self, [p for p in self._params if "initial_connection" not in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(node_dict.keys())) + 1)
        new_id = next(self.node_indexer)
        assert new_id not in node_dict
        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == "true":
            return True
        elif self.structural_mutation_surer == "false":
            return False
        elif self.structural_mutation_surer == "default":
            return self.single_structural_mutation
        else:
            raise RuntimeError("Invalid structural_mutation_surer {!r}".format(self.structural_mutation_surer))




class DefaultGenome:
    def __init__(self, key):
        self.key = key
        self.connections = {}
        self.nodes = {}
        self.fitness = None

    def __str__(self):
        string = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in self.nodes.items():
            string += "\n\t{0} {1!s}".format(k, ng)
        string += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            string += "\n\t" + str(c)
        return string

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = genes.DefaultNodeGene
        param_dict['connection_gene_type'] = genes.DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def configure_new(self, config):
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        if config.n_hid > 0:
            for i in range(config.n_hid):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        if "fs_neat" in config.initial_connection:
            if config.initial_connection == "fs_neat_nohidden":
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == "fs_neat_hidden":
                self.connect_fs_neat_hidden(config)
            else:
                if config.n_hid > 0:
                    print("Warning: initial_connection = fs_neat won't connect to hidden nodes;",
                          "\tif this is desired, set initial_connection = fs_nohidden;",
                          "\tif if not, set initial_connection = fs_neat_hidden", sep="\n", file=sys.stderr)
                self.connect_partial_nodirect(config)

        elif "full" in config.initial_connection:
            if config.initial_connection == "full_nodirect":
                self.connect_full_nodirect(config)
            elif config.initial_connection == "full_direct":
                self.connect_full_direct(config)
            else:
                if config.n_hid > 0:
                    print("Warning: initial_connection = full w/ hid nodes won't do direct inp-outp connections;",
                          "\tif this is desired, set initial_connection = full_nodirect;",
                          "\tif not, set initial_connection = full_direct", sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)

        elif "partial" in config.initial_connection:
            if config.initial_connection == "partial_nodirect":
                self.connect_partial_nodirect(config)
            elif config.initial_connection == "partial_direct":
                self.connect_partial_direct(config)
            else:
                if config.n_hid > 0:
                    print("Warning: initial_connection = partial with hidden nodes will\
                          not do direct input-output connections;",
                          "\tif this is desired, set initial_connection = partial_nodirect {0};".format(
                            config.connection_fraction),
                          "\tif not, set initial_connection = partial_direct {0}".format(
                            config.connection_fraction),
                          sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key) #'.get(key)' instead of '[key]' to return None if '[key]' doesn't exist
            if cg2 is None: # excess or disjoint gene, copy from fittest parent
                self.connections[key] = cg1.copy()
            else: # homologous gene, combine parents' genes
                self.connections[key] = cg1.crossover(cg2)

        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None: #extra gene, copy from fittest parent
                self.nodes[key] = ng1.copy()
            else: #homologous gene, combine parents' genes
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random.random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob + config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection(config)
        else:
            if random.random() < config.node_add_prob:
                self.mutate_add_node(config)
            if random.random() < config.node_delete_prob:
                self.mutate_delete_node(config)
            if random.random() < config.conn_add_prob:
                self.mutate_add_connection(config)
            if random.random() < config.conn_delete_prob:
                self.mutate_delete_connection(config)

        for cg in self.connections.values(): # mutate connection genes
            cg.mutate(config)

        for ng in self.nodes.values(): # mutate node genes
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        conn2split = random.choice(list(self.connections.values()))
        new_n_key = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_n_key)
        self.nodes[new_n_key] = ng

        conn2split.enabled = False
        in_n, out_n = conn2split.key
        # connects in_n node with new node with weight=1.0 & new node with out_n with old weight bewteen in_n and out_n
        self.add_connection(config, in_n, new_n_key, 1.0, True)
        self.add_connection(config, new_n_key, out_n, conn2split.weight, True)

    def add_connection(self, config, in_key, out_key, weight, enabled):
        assert isinstance(in_key, int)
        assert isinstance(out_key, int)
        assert out_key >= 0
        assert isinstance(enabled, bool)

        key = (in_key, out_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_delete_node(self, config):
        available_nodes = [k for k in self.nodes.keys() if k not in config.output_keys]
        if not available_nodes:
            return -1
        del_key = random.choice(available_nodes)
        conn2del = set()
        for key, value in self.connections.items():
            if del_key in value.key:
                conn2del.add(value.key)
        for key in conn2del:
            del self.connections[key]
        del self.nodes[del_key]
        return del_key

    def mutate_add_connection(self, config):
        possible_outputs = list(self.nodes.keys())
        out_node = random.choice(possible_outputs)
        possible_inputs = possible_outputs + config.input_keys
        in_node = random.choice(possible_inputs)

        key = (in_node, out_node)
        # avoids duplicating connections
        if key in self.connections:
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return
        # avoids connecting two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return
        # avoids creating cycles for feed forward networks
        if config.feed_forward and graphs.creates_cycle(list(self.connections.keys()), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_connection(self, config):
        if self.connections:
            key = random.choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes.keys():
                if k2 not in self.nodes:
                    disjoint_nodes += 1
            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + (config.compatibility_disjoint_coefficient * disjoint_nodes)) / max_nodes

        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections.keys():
                if k2 not in self.connections.keys():
                    disjoint_connections += 1
            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    connection_distance += c1.distance(c2, config)

            max_connections = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                  (config.compatibility_disjoint_coefficient *
                                   disjoint_connections)) /max_connections

        distance = node_distance + connection_distance
        return distance

    def size(self):
        n_enabled_conn = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), n_enabled_conn

    @staticmethod
    def create_node(config, node_key):
        node = config.node_gene_type(node_key)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, in_key, out_key):
        connection = config.connection_gene_type((in_key, out_key))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        in_key = random.choice(config.input_keys)
        for out_key in config.output_keys:
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        in_key = random.choice(config.input_keys)
        others = [i for i in self.nodes.keys() if i not in config.input_keys]
        for out_key in others:
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        hidden = [i for i in self.nodes.keys() if i not in config.output_keys]
        output = [i for i in self.nodes.keys() if i in config.output_keys]
        connections = []
        if direct or (not hidden): # direct connections
            for in_key in config.input_keys:
                for out_key in config.output_keys:
                    connections.append((in_key, out_key))
        if hidden: # hidden connections
            for in_key in config.input_keys:
                for h in hidden:
                    connections.append((in_key, h))
            for h in hidden:
                for out_key in config.output_keys:
                    connections.append((h, out_key))
        if not config.feed_forward: # self-connections in RNNs
            for i in self.nodes.keys():
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        for in_key, out_key in self.compute_full_connections(config, False):
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        for in_key, out_key in self.compute_full_connections(config, True):
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        random.shuffle(all_connections)
        n2add = int(round(len(all_connections) * config.connection_fraction))
        for in_key, out_key in all_connections[:n2add]:
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        random.shuffle(all_connections)
        n2add = int(round(len(all_connections) * config.connection_fraction))
        for in_key, out_key in all_connections[:n2add]:
            connection = self.create_connection(config, in_key, out_key)
            self.connections[connection.key] = connection
