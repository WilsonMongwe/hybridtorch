class BaseSampler(object):
  def __init__(self, model, weights_0, sample_size, burn_in_period,
               adapt, target_acceptance):
    self.model = model
    self.weights_0 = weights_0
    self.sample_size = sample_size
    self.burn_in_period = burn_in_period
    self.adapt = adapt
    self.target_acceptance = target_acceptance
    
  def transition(self, w, p):
     raise NotImplementedError()

  def run(self):
     raise NotImplementedError()

