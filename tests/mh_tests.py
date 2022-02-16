import unittest
from samplers.mh import MetropolisHastings as mh
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

sampler = mh(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)
sampler_1 = mh(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size)

class TestMHMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_mh_initialisation(self):
        self.assertEqual(sampler.model.ard, ard)
        self.assertEqual(sampler.model.dimensions, dim)
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertTrue(np.array_equal(sampler.weights_0.detach().numpy(), 
                                       weights.detach().numpy(), equal_nan=True))
      
    def test_mh_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
       
   
    def test_mh_run(self):
        torch.manual_seed(10)
        result = sampler_1.run()

        self.assertEqual(result["no_target_evaluations"], 15)
        self.assertEqual(result["accepted_rate"], 100.0)
        
        print("samples", result["log_like"])
        
        expected_samples = np.array([[ 0.03986072,  0.13986072,  0.23986073],
         [-0.06136026,  0.03863974,  0.13863975],
         [-0.28840208, -0.18840207, -0.08840206],
         [-0.36272803, -0.262728,   -0.16272801],
         [-0.44964606, -0.34964603, -0.24964602]])
        
        expected_log_like = np.array([4.209203,  4.158782,  4.1890407, 4.242026,  4.3308573]) 
        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,)) 
 
if __name__ == '__main__':
    unittest.main()
   