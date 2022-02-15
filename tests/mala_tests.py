import unittest
from samplers.mala import MetropolisAdjustedLengavinAlgorithm as mala
import torch
import numpy as np
from models.blr import LogisticRegression as BLR


# test setup
X = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
Y = torch.tensor([ 1, 0])
dim = 3
ard = False
model = BLR(X, Y, dim, ard)

sample_size = 5
burn_in_period = 2
adapt = False
target_acceptance = 0.8 
step_size = 1e-1
weights = torch.tensor([0.1, 0.2, 0.3])
momentum = torch.tensor([0.11, 0.12, 0.31])

sampler = mala(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)
sampler_1 = mala(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)
sampler_2 = mala(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)
sampler_full = mala(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)

class TestHMCMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_mala_initialisation(self):
        self.assertEqual(sampler.model.ard, ard)
        self.assertEqual(sampler.model.dimensions, dim)
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertEqual(sampler.path_length, 1)
        self.assertTrue(np.array_equal(sampler.weights_0.detach().numpy(), 
                                       weights.detach().numpy(), equal_nan=True))
      
    def test_mala_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
       
   
    def test_mala_transition(self):
        w_result, p_result = sampler_1.transition(weights, momentum)
        
        # expected transition with path_length =1
        dHdw = sampler_1.compute_gradients(model.log_prob(weights), weights)
        p_expected = momentum - 0.5 * sampler_1.step_size * (dHdw)
        
        w_expected = weights + (step_size * p_expected)
        dHdw = sampler_1.compute_gradients(model.log_prob(w_expected), w_expected)
        p_expected = p_expected - 0.5 * sampler_1.step_size * dHdw
        p_expected = - p_expected
        
        self.assertTrue(np.array_equal(w_expected.detach().numpy(), 
                                       w_result.detach().numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(p_expected.detach().numpy(), 
                                       p_result.detach().numpy(), equal_nan=True))
 
  
if __name__ == '__main__':
    unittest.main()
   