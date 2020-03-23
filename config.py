"""Implementation based on NEAT-Python neat-python/neat/config.py:
https://github.com/CodeReclaimers/neat-python/blob/master/neat/config.py"""

import os
import warnings

try:
  from configparser import ConfigParser
except ImportError:
  from ConfigParser import SafeConfigParser as ConfigParser




class ConfigParameter:
    def __init__(self, name, value_type, default=None):
        self.name = name
        self.value_type = value_type
        self.default = default
    
    def __repr__(self):
        if self.default is None:
            return "ConfigParser(%r, %r)" % (self.name, self.value_type)
        return "ConfigParser(%r, %r, %r)" % (self.name, self.value_type, self.default)
  
    def parse(self, section, config_parser):
        if int==self.value_type:
            return config_parser.getint(section, self.name)
        if bool==self.value_type:
            return config_parser.getboolean(section, self.name)
        if float==self.value_type:
            return config_parser.getfloat(section, self.name)
        if list==self.value_type:
            value = config_parser.get(section, self.name)
            return value.split(" ")
        if str==self.value_type:
            return config_parser.get(section, self.name)
    
        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))
    
    def interpret(self, config_dict):
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError("Missing configuraion item: " + self.name)
            else:
                warnings.warn("Using default %r for %s" % (self.default, self.name), DeprecationWarning)
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default
          
        try:
            if str == self.value_type:
                return str(value)
            if int == self.value_type:
                return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type:
                return float(value)
            if list == self.value_type:
                return value.split(" ")
        except Exception:
            raise RuntimeError("Error interpreting config item %s with value %r and type %s" % 
                               (self.name, value, self.value_type))
    
        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))
    
    def format(self, value):
        if list==self.value_type:
            return " ".join(value)
        return str(value)

    
    
  
def write_pretty_params(f, config, params):
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()
    params = dict((p.name, p) for p in params)
    for name in param_names:
        p = params[name]
        f.write('%s = %s\n' % (p.name.ljust(longest_name), p.format(getattr(config, p.name))))

        
        
    
class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass




class DefaultClassConfig:
    def __init__(self, param_dict, param_list):
        self._params = param_list
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" + "\n\t".join(unknown_list))
        raise UnknownConfigItemError("Unknown configuration item %s" % (unknown_list[0]))

    @classmethod
    def write_config(cls, f, config):
        write_pretty_params(f, config, config._params)
    
    
    
   
class Config:
    """A simple container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str), 
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename):
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn("Using default '%r' for '%s'" % 
                                  (p.default, p.name), DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in iterkeys(param_dict) if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError("Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)
      
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write("# The 'NEAT' section specifies parameters particular to the NEAT algorithm\n")
            f.write("# or the experiment itself. This is the only required section.\n")
            f.write("[NEAT]\n")
            write_pretty_params(f, self, self.__params)

            f.write("\n[{0}]\n" % (self.genome_type.__name__))
            self.genome_type.write_config(f, self.genome_config)

            f.write("\n[{0}]\n" % (self.species_set_type.__name__))
            self.species_set_type.write_config(f, self.species_set_config)

            f.write("\n[{0}]\n" % (self.stagnation_type.__name__))
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write("\n[{0}]\n" % (self.reproduction_type.__name__))
            self.reproduction_type.write_config(f, self.reproduction_config)
