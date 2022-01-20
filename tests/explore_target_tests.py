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
        
    def test_exlore_target_run_with_one_sampler(self):
        torch.manual_seed(10)
        explore.run_chains()
        results = explore.results
        
        self.assertEqual(list(results.keys()), ["HMC_0", "HMC_1"])
        
        
        expected_samples_1 = np.array([[-0.0231443 , -0.00689188,  0.23073448],
               [-0.2682258 , -0.16507077,  0.02738051],
               [-0.42919487, -0.23667525,  0.2682095 ],
               [-0.5547742 , -0.36253634,  0.3543284 ],
               [-0.5419699 , -0.13798103,  0.5343509 ]])
        
        
        expected_samples_2 = np.array([[ 0.37533873,  0.17147857,  0.15019979],
         [ 0.50514406,  0.39861268, -0.13994624],
         [ 0.20638163,  0.6104319 , -0.3933664 ],
         [ 0.65615755,  0.6360218 , -0.23652095],
         [ 0.7448362 ,  0.5825212 , -0.34160024]])
        
        
        expected_log_like_1 = np.array([4.181287 , 4.171888 , 4.273247 , 4.391862 , 4.4549127]) 
        
        expected_log_like_2 = np.array([4.280509 , 4.405637 , 4.4876223, 4.6547217, 4.7210464]) 

                                       
        self.assertTrue(np.allclose(results["HMC_0"]["samples"], 
                                    expected_samples_1, rtol = 1e-7, equal_nan=True,))
        self.assertTrue(np.allclose(results["HMC_1"]["samples"], 
                                    expected_samples_2, rtol = 1e-7, equal_nan=True,))
        
        self.assertTrue(np.allclose(results["HMC_0"]["log_like"], 
                                    expected_log_like_1, rtol = 1e-7, equal_nan=True,))
        
        self.assertTrue(np.allclose(results["HMC_1"]["log_like"], 
                                    expected_log_like_2, rtol = 1e-7, equal_nan=True,))
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
   