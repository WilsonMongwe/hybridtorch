from samplers.base_sampler import BaseSampler

class HamiltonianMonteCarlo(BaseSampler):
  def __init__(self, model,  weights_0, sample_size, burn_in_period,
               adapt, target_acceptance, step_size, path_length):
      BaseSampler.__init__(self, model, weights_0, sample_size, burn_in_period,
                           adapt, target_acceptance)
      self.step_size = step_size
      self.path_length = path_length
    
  def transition(self, w_current_sate):
     raise NotImplementedError()

  def run(self):
     raise NotImplementedError()

