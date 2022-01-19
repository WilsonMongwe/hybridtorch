import unittest
from samplers.hmc import HamiltonianMonteCarlo as HMC


# test setup
sample_size = 5
burn_in_period = 2
adapt = True
target_acceptance = 0.8 
step_size = 1e-1
path_length = 5

sampler = HMC(sample_size, burn_in_period, adapt, target_acceptance, step_size, path_length)

class TestHMCMethods(unittest.TestCase):
    
    def test_hmc_initialisation(self):
        self.assertEqual(sampler.adapt, adapt)
        self.assertEqual(sampler.sample_size, sample_size)
        self.assertEqual(sampler.burn_in_period, burn_in_period)
        self.assertEqual(sampler.target_acceptance, target_acceptance)
        self.assertEqual(sampler.step_size, step_size)
        self.assertEqual(sampler.path_length, path_length)
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
   