from samplers.hmc import HamiltonianMonteCarlo
from torch.autograd import grad
from utilities.utils import leapfrog_intergrator as leap
from utilities.utils import metropolis_acceptance_step as mh_step
from utilities.utils import adaptation
from utilities.utils import adaptation_params
from tqdm import tqdm

import time
import numpy as np
import torch

class SeperableShadowHybridMonteCarlo(HamiltonianMonteCarlo):
  def __init__(self, model,  weights_0, sample_size, burn_in_period,
               adapt, target_acceptance, step_size, path_length):
      HamiltonianMonteCarlo.__init__(self, model,  weights_0, sample_size, burn_in_period,
                   adapt, target_acceptance, step_size, path_length)
      self.tolerance = 1e-6
      self.max_iterations = 100
      
  def target(self, weights):
      self.no_target_evaluations += 1
      return self.model.log_prob(weights)
  
  def kinetic(self, momentum):
      return 0.5 * torch.matmul(momentum, momentum.T)
  
  def hamiltonian(self, weights, momentum):
      return self.kinetic(momentum) + self.target(weights)
    
  def shadow_hamiltonian(self, weights, p):
      # Compute shadow: S(x,p) = H(x,p) + dt2 / 24 UT_x M-1 U_x
      loss = self.target(weights)
      force = self.compute_gradients(loss,weights)
      dt2d24 = self.step_size * self.step_size / 24.0
      H_x_p = self.hamiltonian(weights, p) 
      Uq_M_Uq = torch.matmul(force.T,force)
      shadow  = H_x_p + dt2d24 * Uq_M_Uq
      return shadow
  
  def shadow_target(self, weights):
      loss = self.target(weights)
      force = self.compute_gradients(loss,weights)
      dt2d24 = self.step_size * self.step_size/24.0
      Uq_M_Uq = torch.matmul(force.T,force)

      return self.target(weights) + dt2d24 * Uq_M_Uq
  
  def solve_for_p_hat(self, dt, dt_24, weights, momentum):
      new_p_hat = momentum
      norm = 1
      count = 0
      while (norm > self.tolerance):
          prev_p_hat = new_p_hat
          
          forward = weights + dt * prev_p_hat
          backward = weights - dt * prev_p_hat
          loss_forward = self.target(forward)
          loss_backward = self.target(backward)
            
          U_x_forward = self.compute_gradients(loss_forward, forward)
          U_x_back = self.compute_gradients(loss_backward, backward)
          
          new_p_hat = momentum - dt_24 * (U_x_forward - U_x_back)
          norm = torch.norm(new_p_hat - prev_p_hat)
          count +=1
          if (count > self.max_iterations):
              break
      return new_p_hat
    
  def pre_processing(self, momentum, weights):
        dt = self.step_size
        dt_24 = dt / 24.0
        dt2_24 = dt * dt_24
        
        new_p_hat = self.solve_for_p_hat(dt, dt_24, weights, momentum)
        
        forward = weights + dt * new_p_hat
        backward = weights - dt * new_p_hat
        loss_forward = self.target(forward)
        loss_backward = self.target(backward)
            
        U_x_forward = self.compute_gradients(loss_forward, forward)
        U_x_back = self.compute_gradients(loss_backward, backward)
        
        weights_hat = weights + dt2_24 * (U_x_forward + U_x_back) 

        return new_p_hat, weights_hat
    
  def solve_for_weights_hat(self, weights_hat, momentum_hat, dt, dt2_24, U_x):
        new_weights = weights_hat
        norm = 1
        count = 0
        while (norm > self.tolerance):
            prev_weights = new_weights
            
            forward = prev_weights + dt * momentum_hat
            backward = prev_weights - dt * momentum_hat
            loss_forward = self.target(forward)
            loss_backward = self.target(backward)
            
            U_x_back = self.compute_gradients(loss_forward, forward)
            U_x_forward = self.compute_gradients(loss_backward, backward)
        
            new_weights = weights_hat -  dt2_24 * (U_x_back + U_x_forward) 
          
            norm = torch.norm(new_weights - prev_weights,2)
            count += 1
            if (count > self.max_iterations):
                break
        return new_weights

  def post_processing(self, momentum_hat, weights_hat):
        dt = self.step_size
        loss = self.target(weights_hat)
        U_x = self.compute_gradients(loss, weights_hat)
        U_x = U_x.T
 
        dt_24 = dt / 24.0
        dt2_24 = dt * dt_24
       
        new_weights = self.solve_for_weights_hat(weights_hat, momentum_hat, dt, dt2_24, U_x)
        
        forward = new_weights + dt * momentum_hat
        backward = new_weights - dt * momentum_hat
        loss_forward = self.target(forward)
        loss_backward = self.target(backward)
            
        U_x_back = self.compute_gradients(loss_forward, forward)
        U_x_forward = self.compute_gradients(loss_backward, backward)
          
        momentum = momentum_hat + dt_24 * (U_x_forward - U_x_back)

        return momentum, new_weights    
     
  def compute_gradients(self,log_p_eval_at_w, w):
      self.no_grad_evaluations += 1
      return grad(log_p_eval_at_w, w)[0]
    
  def transition(self, weights, momentum):
      weights, momentum = leap(weights, momentum, self.step_size, 
                               self.path_length, self.shadow_target, 
                               self.compute_gradients)
      return weights, momentum

  def run(self):
      print(":::::::::: Running S2HMC :::::::::::")
  
      # initialise variables      
      samples = []
      log_lik = []
      weights = self.weights_0
      n_accepted = 0
      importance_weights = [] #weighting for observables
      
      # initialise adaptation parameters
      Hbar_old, eps_bar, mu = adaptation_params(self.step_size)

      for i in tqdm(range(self.sample_size)): 
          
          momentum = torch.normal(0,1,(self.dim,))
          
          old_shadow_hamiltonian = self.shadow_hamiltonian(weights, momentum)
          
          # pre processing step
          momentum_hat, weights_hat = self.pre_processing(momentum, weights)
          
          new_weights, new_momentum = self.transition(weights_hat, momentum_hat)
          
          # post processing step
          new_momentum , new_weights = self.post_processing(new_momentum, new_weights)
          
          new_shadow_hamiltonian = self.shadow_hamiltonian(new_weights, new_momentum)
             
          new_Hamiltonian = self.hamiltonian(new_weights, new_momentum)
          
          # store importance weights
          importance_weights.append((torch.exp(new_shadow_hamiltonian 
                                               - new_Hamiltonian)).detach().clone().numpy())
             
          u = torch.rand(())
          accept_reject, alpha = mh_step(u,new_shadow_hamiltonian, old_shadow_hamiltonian, i) 
    
          if accept_reject :
             weights = new_weights.detach().clone()
             n_accepted +=1
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
    
    
      end_time = time.time()
      total_time = (end_time - start_time)
  
      accepted_rate = (n_accepted / self.sample_size)*100
      print("\n")
      print("Total time :", total_time)  
      print("Acceptance Rate: ", accepted_rate, "%")
      print('Final Step Size: ', self.step_size )     
      
      results = {
                  "samples": np.asarray(samples),
                  "log_like": np.asarray(log_lik),
                  "importance_weights": np.asarray(importance_weights),
                  "step_size": self.step_size,
                  "total_time": total_time,
                  "accepted_rate": accepted_rate,
                  "no_grad_evaluations": self.no_grad_evaluations,
                  "no_target_evaluations": self.no_target_evaluations,
                }
      
      self.no_grad_evaluations = 0
      self.no_target_evaluations = 0
      print(":::::::::: S2HMC Finished:::::::::::")    
      return results
    
