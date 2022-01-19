import unittest
from samplers.hmc import HamiltonianMonteCarlo as HMC
import torch
import numpy as np
from models.logistic_regression import LogisticRegression as BLR


# test setup
X = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
Y = torch.tensor([ 1, 0])
dim = 3
ard = True
model = BLR(X, Y, dim, ard)

sample_size = 5
burn_in_period = 2
adapt = True
target_acceptance = 0.8 
step_size = 1e-1
path_length = 5
weights = torch.tensor([0.1, 0.2, 0.3])

sampler = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)

class TestHMCMethods(unittest.TestCase):
    
    def test_hmc_initialisation(self):
        self.assertEqual(sampler.model.ard, ard)
        self.assertEqual(sampler.model.dimensions, dim)
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertEqual(sampler.path_length, path_length)
        self.assertTrue(np.array_equal(sampler.weights_0.detach().numpy(), 
                                       weights.detach().numpy(), equal_nan=True))

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
   