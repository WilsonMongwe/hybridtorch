import unittest
from samplers.hmc import HamiltonianMonteCarlo as HMC
import torch
import numpy as np
from models.blr import LogisticRegression as BLR
from utilities.explore_target import ExploreTarget


# test setup
#model
X = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
Y = torch.tensor([ 1, 0])
dim = 3
ard = False
model = BLR(X, Y, dim, ard)

#sampler
sample_size = 5
burn_in_period = 2
adapt = False
target_acceptance = 0.8 
step_size = 1e-1
path_length = 2
weights = torch.tensor([0.1, 0.2, 0.3])
momentum = torch.tensor([0.11, 0.12, 0.31])
sampler = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)

#explre target
chains = 2
explore = ExploreTarget(model,["HMC"], [sampler], chains)


class TestExploreTargetMethods(unittest.TestCase):
    
    def test_explore_target_initialisation(self):
        self.assertEqual(explore.number_of_chains, chains)
        self.assertEqual(explore.number_of_samplers, 1)

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
   