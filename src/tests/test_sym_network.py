import os
import sys
import io

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
sys.path.append(os.getcwd())

import torch as th
from src.mygym.networks.simple_sym_network import SetEncodeProcessDecode

# input: shape(N, m, di)
# latent: shape(N, m, dl)
DEFUALT_INPUT_DIM = 4 # di
DEFAULT_LATENT_DIM = 8 # dl
DEFAULT_INPUT_NUM = 2 # m
DEFAULT_N_ENV = 4 # N



class NetworkSymmetryChecker():
    def __init__(self, 
                 network: th.nn.Module,
                 input_dim = DEFUALT_INPUT_DIM,
                 latent_dim = DEFAULT_LATENT_DIM,
                 num_env = DEFAULT_N_ENV,
                 num_input = DEFAULT_INPUT_NUM):
        self.network = network
        self.di = input_dim
        self.dl = latent_dim
        self.N_env =num_env
        self.m = num_input
        
    def _generate_random_input(self):
        return th.rand(self.N_env, self.m, self.di)
    
    def _permutate_element(self, 
                           input: th.Tensor): # n,m,di 
        # permutate 2nd axis, e.g. x(:,[0,1],:) = x(:,[1,0],:)
        permutated = input[:,[1,0],:].clone()
        return permutated
        
    
    def check_symmetry(self):
        input = self._generate_random_input()
        input_sym = self._permutate_element(input)

        output = self.network(input)
        output_sym = self.network(input_sym)
        output_sym_sym = self._permutate_element(output_sym)

        # print(output)
        # print(output_sym)
        # print(output_sym_sym)

        diff = output - output_sym_sym
        err = diff.norm()
        print(err)




if __name__ == "__main__":
    sym_net = SetEncodeProcessDecode(
                DEFUALT_INPUT_DIM, DEFAULT_LATENT_DIM)
    checker = NetworkSymmetryChecker(network=sym_net)
    checker.check_symmetry()