from samplers.hmc import HamiltonianMonteCarlo

# MALA is HMC with path length = 1
class MetropolisAdjustedLengavinAlgorithm(HamiltonianMonteCarlo):
  def __init__(self, model,  weights_0, sample_size, burn_in_period,
               adapt, target_acceptance, step_size):
      HamiltonianMonteCarlo.__init__(self, model,  weights_0, sample_size, burn_in_period,
                   adapt, target_acceptance, step_size, 1)
