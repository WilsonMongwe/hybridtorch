import torch

class ExploreTarget(object):
  def __init__(self, model, sampler_names, samplers, number_of_chains):
      self.model = model
      self.sampler_names = sampler_names
      self.samplers = samplers
      self.number_of_chains = number_of_chains
      self.number_of_samplers = len(self.sampler_names)
      self.initial_states = torch.randn(self.number_of_chains, model.dimensions)
      self.results = {}
      
  def run_chains(self):
      for chain in range (self.number_of_chains):
          for s in range(self.number_of_samplers):
              name = self.sampler_names[s].upper()
              print( name+ "  -----> ",  chain)
              self.results[name + "_" + str(chain)] = self.samplers[s].run()
              
             
          

