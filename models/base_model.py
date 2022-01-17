class BaseModel():
  def __init__(self, dimensions, ard):
    self.dimensions = dimensions
    self.ard = ard
    
  def predictions(self, X, w):
     raise NotImplementedError()

  def log_prob(self, w):
     raise NotImplementedError()

