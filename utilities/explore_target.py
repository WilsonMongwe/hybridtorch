import torch

class ExploreTarget(object):
  def __init__(self, sampler_names, samplers, number_of_chains):
      if (number_of_chains <= 0):
          raise Exception("number_of_chains must be greater than zero")
      self.number_of_samplers = len(sampler_names)
      if (self.number_of_samplers == 0):
          raise Exception("samplers list must have atleast one element")
      
      self.model = samplers[0].model
      self.sampler_names = sampler_names
      self.samplers = samplers
      self.number_of_chains = number_of_chains
      self.initial_states = torch.randn(self.number_of_chains, self.model.dimensions)
      self.results = {}
      
  def run_chains(self):
      for chain in range (self.number_of_chains):
          for s in range(self.number_of_samplers):
              name = self.sampler_names[s].upper()
              print( name+ "  -----> ",  chain)
              self.results[name + "_" + str(chain)] = self.samplers[s].run()
              
  def ess(self):
      return

  def r_hat(self):
      return
  
  def mean_likelihood(self, x_test = 0, y_test = 0):
      return
  
  def plot_ard_importances(self):
      # ensure that the samplers return the samples and alphas seperately
      return
  
  def predictive_performnce(self, x_test, y_test):
      # mse, auc etc
      return

  def plot_log_prob(self):
      # likelihoods for different samplers
      return
  
  def plot_ess(self):
      # ess over time, grad val, func val
      return
  
  def plot_r_hat(self):
      return
          
          

