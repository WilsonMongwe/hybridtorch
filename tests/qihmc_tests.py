import unittest
from samplers.qihmc import QuantumInspiredHamiltonianMonteCarlo as qihmc
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
path_length = 2
weights = torch.tensor([0.1, 0.2, 0.3])
momentum = torch.tensor([0.11, 0.12, 0.31])
vol_of_vol = 0.1

sampler = qihmc(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length, vol_of_vol)
sampler_1 = qihmc(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, 1, vol_of_vol)
sampler_2 = qihmc(model, weights, sample_size, burn_in_period,
                  adapt, target_acceptance, step_size, 3, vol_of_vol)

class TestQIHMCMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_QIhmc_initialisation(self):
        self.assertEqual(sampler.vol_of_vol, vol_of_vol)
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
        
    def test_QIhmc_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
        
         
    def test_hmc_run(self):
        torch.manual_seed(10)
        result = sampler_2.run()

        self.assertEqual(result["no_grad_evaluations"], 20)
        self.assertEqual(result["no_target_evaluations"], 35)
        self.assertEqual(result["accepted_rate"], 100.0)
        
        expected_samples = np.array([[-0.29398662,  0.4854516,   0.17234404],
         [-1.0300815,   0.18807721,  0.22696412],
         [-0.94604933,  0.20839296, -0.20626129],
         [-1.3780799,   0.70243156, -0.10305298],
         [-1.2747024,   0.85973334,  0.02424147]])
        
        expected_log_like = np.array([4.3386927, 4.677961,  4.6220527, 5.321604 , 5.3051276]) 
        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,))                     
if __name__ == '__main__':
    unittest.main()
   