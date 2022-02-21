import unittest
from samplers.qimhmc import QuantumInspiredMagneticHamiltonianMonteCarlo as qimhmc
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
vol_of_vol = 0.1

G = torch.tensor([[0.0, 0.1, 0.1], [0.1, 0.2, 0.0] , [0.0, 0.1, 0.1]])
G = (G -G.T)/ 2

sampler = qimhmc(model, weights, sample_size, burn_in_period, adapt, 
               target_acceptance, step_size, path_lehth, G, vol_of_vol)
sampler_1 = qimhmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, path_lehth, G, vol_of_vol)

sampler_2 = qimhmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, 3, G, vol_of_vol)

class TestMHMCMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_qimhmc_initialisation(self):
        self.assertEqual(sampler.model.ard, ard)
        self.assertEqual(sampler.vol_of_vol, vol_of_vol)
        self.assertEqual(sampler.model.dimensions, dim)
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertEqual(sampler.path_length, path_lehth)
        self.assertTrue(np.array_equal(sampler.weights_0.detach().numpy(), 
                                       weights.detach().numpy(), equal_nan=True))
      
    def test_mhmc_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
    
    def test_mhmc_run(self):
        torch.manual_seed(10)
        result = sampler_2.run()

        self.assertEqual(result["no_grad_evaluations"], 20)
        self.assertEqual(result["no_target_evaluations"], 35)
        self.assertEqual(result["accepted_rate"], 80.0)
        
        expected_samples = np.array([[-0.29503828,  0.4865986,   0.16740763],
         [-1.0304097,   0.18881159,  0.22159706],
         [-0.94913715,  0.21243344, -0.21069859],
         [-0.94913715,  0.21243344, -0.21069859],
         [-0.86083096,  0.3840326 , -0.07922661]])
        
        expected_log_like = np.array([4.3382387, 4.6769934, 4.6274796, 4.6274796 , 4.572481 ]) 
        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,))  
        
  
if __name__ == '__main__':
    unittest.main()
   