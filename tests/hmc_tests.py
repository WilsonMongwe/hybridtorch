import unittest
from samplers.hmc import HamiltonianMonteCarlo as HMC
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

sampler = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)
sampler_1 = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, 1)
sampler_2 = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)
sampler_full = HMC(model, weights, sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)

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
        
    def test_hmc_target(self):
        sampler_1.target(weights)
        sampler_1.target(weights)
        self.assertEqual(sampler_1.no_target_evaluations, 2)
        
    def test_hmc_kinetic(self):
        result = sampler.kinetic(momentum).numpy()
        expected = 0.5 * torch.matmul(momentum, momentum.T).numpy()
        self.assertEqual(result, expected)
        
    def test_hmc_hamiltonian(self):
        result = sampler.hamiltonian(weights, momentum)
        expected = sampler.target(weights) +  sampler.kinetic(momentum)
        self.assertEqual(result.detach().numpy(), expected.detach().numpy())
        
    def test_hmc_gradients(self):
        sampler_2.compute_gradients(model.log_prob(weights), weights)
        self.assertEqual(sampler_2.no_grad_evaluations, 1)
        
    def test_hmc_transition_with_unit_path_length(self):
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
        
    def test_hmc_transition_general(self):
         w_result, p_result = sampler.transition(weights, momentum)
         
         # expected transition with path_length =1
         dHdw = sampler.compute_gradients(model.log_prob(weights), weights)
         p_expected = momentum - 0.5 * sampler.step_size * (dHdw)
         
         w_expected = weights + (step_size * p_expected)
         dHdw = sampler.compute_gradients(model.log_prob(w_expected), w_expected)
         p_expected = p_expected - sampler.step_size * dHdw
         
         w_expected = w_expected + (step_size * p_expected)
         dHdw = sampler.compute_gradients(model.log_prob(w_expected), w_expected)
         p_expected = p_expected - 0.5 * sampler.step_size * dHdw
         p_expected = - p_expected
         
         self.assertTrue(np.allclose(w_expected.detach().numpy(), 
                                     w_result.detach().numpy(), rtol = 1e-8, equal_nan=True,))
         self.assertTrue(np.allclose(p_result.detach().numpy(), 
                                     p_expected.detach().numpy(), rtol = 1e-8, equal_nan=True,))
         
    def test_hmc_run(self):
        torch.manual_seed(10)
        result = sampler_full.run()

        self.assertEqual(result["no_grad_evaluations"], 15)
        self.assertEqual(result["no_target_evaluations"], 30)
        self.assertEqual(result["accepted_rate"], 100.0)
        
        expected_samples = np.array([[-0.0231443 , -0.00689188,  0.23073448],
               [-0.2682258 , -0.16507077,  0.02738051],
               [-0.42919487, -0.23667525,  0.2682095 ],
               [-0.5547742 , -0.36253634,  0.3543284 ],
               [-0.5419699 , -0.13798103,  0.5343509 ]])
        
        expected_log_like = np.array([4.181287 , 4.171888 , 4.273247 , 4.391862 , 4.4549127]) 
        
        self.assertTrue(np.allclose(result["samples"], 
                                    expected_samples, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(result["log_like"], 
                                    expected_log_like, rtol = 1e-7, equal_nan=True,))
        

    

        
if __name__ == '__main__':
    unittest.main(verbosity=2)
   