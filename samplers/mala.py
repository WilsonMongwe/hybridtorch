from samplers.hmc import HamiltonianMonteCarlo

class MetropolisAdjustedLengavinAlgorithm(HamiltonianMonteCarlo):
  # MALA is HMC with path length = 1
  def __init__(self, model,  weights_0, sample_size, burn_in_period,
               adapt, target_acceptance, step_size):
      HamiltonianMonteCarlo.__init__(self, model,  weights_0, sample_size, burn_in_period,
                   adapt, target_acceptance, step_size, 1)
