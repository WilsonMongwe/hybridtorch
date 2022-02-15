import unittest
from samplers.mhmc import MagneticHamiltonianMonteCarlo as mhmc
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

sampler = mhmc(model, weights, sample_size, burn_in_period, adapt, 
               target_acceptance, step_size, path_lehth, G)
sampler_1 = mhmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, path_lehth, G)

sampler_2 = mhmc(model, weights, sample_size, burn_in_period, adapt, 
                 target_acceptance, step_size, 3, G)

class TestHMCMethods(unittest.TestCase):
    
    def setUp(self):
        print ("In method", self._testMethodName)
    
    def test_mhmc_initialisation(self):
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
      
    def test_mhmc_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
        
   
    def test_mhmc_transition(self):
        w_result, p_result, G_result = sampler_1.transition(weights, momentum, G)
        
        # expected transition with path_length =1
        
        G_exp, factor = magnetic_field_exp_and_factor(G, step_size)

        dHdw = sampler_1.compute_gradients(model.log_prob(weights), weights)
        p_expected = momentum - 0.5 * sampler_1.step_size * (dHdw)
        
        w_expected = weights + torch.matmul(factor , p_expected)
        p_expected = torch.matmul(G_exp, p_expected)
        
        dHdw = sampler_1.compute_gradients(model.log_prob(w_expected), w_expected)
        p_expected = p_expected - 0.5 * sampler_1.step_size * dHdw
        p_expected = - p_expected
        
        self.assertTrue(np.array_equal(w_expected.detach().numpy(), 
                                       w_result.detach().numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(p_expected.detach().numpy(), 
                                       p_result.detach().numpy(), equal_nan=True))
        
        self.assertTrue(np.array_equal(-G.detach().numpy(), 
                                       G_result.detach().numpy(), equal_nan=True))
        
        
    def test_mhmc_run(self):
        torch.manual_seed(10)
        result = sampler_2.run()

        self.assertEqual(result["no_grad_evaluations"], 20)
        self.assertEqual(result["no_target_evaluations"], 35)
        self.assertEqual(result["accepted_rate"], 80.0)
        
        expected_samples = np.array([[ 0.16226351,  0.13773648 , 0.19105917],
         [ 0.09457321 , 0.2054268 , -0.11499383],
         [ 0.0317698,   0.2682302  , 0.24756917],
         [ 0.03535197 , 0.26464802 , 0.3679263 ],
         [ 0.03535197,  0.26464802 , 0.3679263 ]])
        
        expected_log_like = np.array([4.212966 , 4.191574 , 4.246612 , 4.3038206 ,4.3038206]) 
        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,))  
        
  
if __name__ == '__main__':
    unittest.main()
   