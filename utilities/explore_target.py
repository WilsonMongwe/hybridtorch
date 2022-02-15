import torch
from utilities.utils import multiESS as mESS
import arviz as az
import numpy as np


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
              
              
  def ess(self, ess_method = "multivariate", mESS_type='less'):
      
      if (len(self.results.keys()) == 0):
          raise Exception('ess: Results object is empty. run the run_chains method first')       
      
      if (ess_method == "multivariate"):
          for s in range(self.number_of_samplers):      
              multi_ess = []           
              for chain in range(self.number_of_chains):
                  name = self.sampler_names[s].upper() + "_" + str(chain)
                  samples = self.results[name]["samples"]
                  multi_ess.append(mESS(samples, b = mESS_type))
                      
              self.results[self.sampler_names[s].upper() 
                           + "_"+"multivariate"] = torch.tensor(multi_ess)
              
      if (ess_method == "univariate"):
           raise Exception('Chosen ESS calc method not supported.')
           
           

  def r_hat(self):
      if (len(self.results.keys()) == 0):
          raise Exception('r_hat: Results object is empty. run the run_chains method first') 
      
      max_rhat = []
      for s in range(self.number_of_samplers): 
          samples = []
          for chain in range(self.number_of_chains):
              name = self.sampler_names[s].upper() + "_" + str(chain)
              samples.append(self.results[name]["samples"])
         
          idata = az.convert_to_inference_data(np.array(samples))
          rhat_min_for_sampler = az.rhat(idata).max().x.values.reshape(1,1)[0]
          max_rhat.append(rhat_min_for_sampler[0])
        
          self.results[self.sampler_names[s].upper() 
                     + "_"+"r_hat"] = torch.tensor(max_rhat)          


          
          

