import torch
from utilities.utils import multiESS as mESS

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
              
  def ess(self, ess_method = "multivariate"):
      
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first')       
      
      if (ess_method == "multivariate"):
          for s in range(self.number_of_samplers):      
              multi_ess = []           
              for chain in range(self.number_of_chains):
                  name = self.sampler_names[s].upper() + "_" + str(chain)
                  samples = self.results[name]["samples"]
                  multi_ess.append(mESS(samples, b='less'))
                      
              self.results[self.sampler_names[s].upper() 
                           + "_"+"multivariate"] = torch.tensor(multi_ess)
              
      if (ess_method == "univariate"):
           raise Exception('Chosen ESS calc method not supported.')

  def r_hat(self):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      return
  
  def mean_likelihood(self, x_test = 0, y_test = 0):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      return
  
  def plot_ard_importances(self):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      # ensure that the samplers return the samples and alphas seperately
      return
  
  def predictive_performnce(self, x_test, y_test):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      # mse, auc etc
      return

  def plot_log_prob(self):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      # likelihoods for different samplers
      return
  
  def plot_ess(self):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      # ess over time, grad val, func val
      return
  
  def plot_r_hat(self):
      if (len(self.results.keys()) == 0):
          raise Exception('Results object is empty. run the run_chains method first') 
      return
          
          

