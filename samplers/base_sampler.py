class BaseSampler(object):
  def __init__(self, sample_size, burn_in_period,
               adapt, target_acceptance):
    self.sample_size = sample_size
    self.burn_in_period = burn_in_period
    self.adapt = adapt
    self.target_acceptance = target_acceptance
    
  def transition(self, w_current_sate):
     raise NotImplementedError()

  def run(self):
     raise NotImplementedError()

