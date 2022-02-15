from samplers.hmc import HamiltonianMonteCarlo
from utilities.utils import magnetic_leap_frog_intergrator as leap
from utilities.utils import metropolis_acceptance_step as mh_step
from utilities.utils import adaptation
from utilities.utils import adaptation_params
from tqdm import tqdm

import time
import numpy as np
import torch

# Magnetic Hmailtonian Monte Carlo
class MagneticHamiltonianMonteCarlo(HamiltonianMonteCarlo):
  def __init__(self, model,  weights_0, sample_size, burn_in_period,
               adapt, target_acceptance, step_size, path_length, magnetic_field):
      HamiltonianMonteCarlo.__init__(self, model,  weights_0, sample_size, burn_in_period,
                   adapt, target_acceptance, step_size, path_length)
      
      self.G = magnetic_field

    
  def transition(self, weights, momentum, G):
      weights, momentum, G = leap(weights, momentum, self.step_size, 
                               self.path_length, self.target, 
                               self.compute_gradients, G)
      return weights, momentum, G

  def run(self):
      print(":::::::::: Running MHMC :::::::::::")
  
      # initialise variables      
      samples = []
      log_lik = []
      weights = self.weights_0
      n_accepted = 0
      
      # initialise adaptation parameters
      Hbar_old, eps_bar, mu = adaptation_params(self.step_size)

      for i in tqdm(range(self.sample_size)): 
          momentum = torch.normal(0,1,(self.dim,))
          
          old_Hamiltonian = self.hamiltonian(weights, momentum)  
             
          new_weights, new_momentum, new_G = self.transition(weights, momentum, self.G)
             
          new_Hamiltonian = self.hamiltonian(new_weights, new_momentum)
             
          u = torch.rand(())
          accept_reject, alpha = mh_step(u,new_Hamiltonian, old_Hamiltonian, i)  
    
          if accept_reject :
             weights = new_weights.detach().clone()
             n_accepted += 1
             self.G = new_G
             samples.append(weights.detach().clone().numpy())
             log_lik.append(self.target(weights).detach().clone().numpy())
    
          else:
              samples.append(weights.detach().clone().numpy())
              log_lik.append(self.target(weights).detach().clone().numpy())
      
          if self.adapt:
              if i < self.burn_in_period:
                 self.step_size, eps_bar, Hbar_old = adaptation(i, mu, Hbar_old, alpha.detach().numpy(), 
                                                                   eps_bar, self.target_acceptance)
              else:
                 self.step_size = eps_bar 
                     
          if i == self.burn_in_period:
              start_time = time.time()
    
          self.G = -new_G
          
      end_time = time.time()
      total_time = (end_time - start_time)
  
      accepted_rate = (n_accepted / self.sample_size) * 100
      print("\n")
      print("Total time :", total_time)  
      print("Acceptance Rate: ", accepted_rate, "%")
      print('Final Step Size: ', self.step_size )     
      
      results = {
                  "samples": np.asarray(samples),
                  "log_like": np.asarray(log_lik),
                  "step_size": self.step_size,
                  "total_time": total_time,
                  "accepted_rate": accepted_rate,
                  "no_grad_evaluations": self.no_grad_evaluations,
                  "no_target_evaluations": self.no_target_evaluations,
                }
      
      self.no_grad_evaluations = 0
      self.no_target_evaluations = 0
      print(":::::::::: MHMC Finished:::::::::::")    
      return results
    
