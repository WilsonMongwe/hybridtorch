import unittest
from samplers.s2hmc import SeperableShadowHybridMonteCarlo as s2hmc
import torch
import numpy as np
from models.blr import LogisticRegression as BLR
from utilities.utils import magnetic_field_exp_and_factor


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
path_lehth = 1

G = torch.tensor([[0.0, 0.1, 0.1], [0.1, 0.2, 0.0] , [0.0, 0.1, 0.1]])
G = (G -G.T)/ 2

sampler = s2hmc(model, weights, sample_size, burn_in_period, adapt, 
               target_acceptance, step_size, path_lehth)
sampler_1 = s2hmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, path_lehth)

sampler_2 = s2hmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, 3)

class TestS2HMCMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_s2hmc_initialisation(self):
        self.assertEqual(sampler.model.ard, ard)
        self.assertEqual(sampler.model.dimensions, dim)
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertEqual(sampler.path_length, path_lehth)
        self.assertTrue(np.array_equal(sampler.weights_0.detach().numpy(), 
                                       weights.detach().numpy(), equal_nan=True))
        self.assertEqual(sampler.tolerance, 1e-6)
        self.assertEqual(sampler.max_iterations, 100)
      
    def test_s2hmc_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
        
   
    def test_s2hmc_transition(self):
        return
        
        
    def test_s2hmc_run(self):
        torch.manual_seed(10)
        result = sampler_2.run()

        self.assertEqual(result["no_grad_evaluations"], 123)
        self.assertEqual(result["no_target_evaluations"], 143)
        self.assertEqual(result["accepted_rate"], 100.0)
        
        expected_samples = np.array([[-0.08539007, -0.11126358,  0.1908932 ],
         [-0.4466915,  -0.34187567, -0.11110405],
         [-0.6739865 , -0.43809247,  0.2552854 ],
         [-0.84375435, -0.6131196 ,  0.3798303 ],
         [-0.8044727,  -0.2639995 ,  0.6386736 ]])
        
        expected_log_like = np.array([4.167862,  4.2815695, 4.44591,   4.6941357, 4.709428 ]) 
        
        expected_weights = np.array([1.000032,  1.000133,  1.000258,  1.0004717, 1.0005475]) 

        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,)) 
        self.assertTrue(np.allclose(result["importance_weights"], 
                                    expected_weights, rtol = 1e-7, equal_nan=True,))
        
  
if __name__ == '__main__':
    unittest.main()
   