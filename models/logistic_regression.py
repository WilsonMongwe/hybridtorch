from models.base_model import BaseModel
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

class LogisticRegression(BaseModel):
  # constants
  ALPHA = 1 # variance of prior distribution 
  prior = Normal(loc = 0, scale = ALPHA , validate_args= False)
  JITTER = 0.001 # to prevent division by zero when using ARD
  
  def __init__(self, X, Y, dimensions, ard):
      BaseModel.__init__(self, dimensions, ard) 
      self.X = X # X is assumed to not include the bias term
      self.feature_size = self.X.shape[1]   
      # append the intercept term to X
      self.adjustedX = torch.cat((self.X, torch.ones(self.feature_size,1)), 1)
      self.Y = Y
      # When perfoming ARD the parameter space is doubled
      # Half of the dimension is the paramaters and the rest the alphas
      self.num_params = self.dimensions
      if (self.ard):
          self.num_params =  self.num_params * 2
          self.num_params_half = self.num_params / 2
      
  def unflattern(self, w):
      if (self.ard):
          w = w[0:self.num_params_half]
            
      weights = w[0:self.feature_size]
      bias = w[self.feature_size:self.num_params]
      return weights.reshape(self.feature_size, ), bias  
    
  def predictions(self, X, w):
      weights, bias = self.unflattern(w)
      outputs = torch.matmul(X.double(), weights.double()) + bias
      out = nn.Sigmoid(outputs.reshape(outputs.shape[0], 1))
      return out

  def log_prob(self, w):
      w.requires_grad_(True)
      if (self.ard):
          # model parameters
          w_param = w[0:self.num_params_half]
          # variance of prior distribution
          w_alphas = torch.exp(w[self.num_params_half:self.num_params])**2
          Xw = torch.matmul(self.adjustedX.double(),w_param.double())
        
          term_1 =  torch.sum(self.Y * nn.LogSigmoid(Xw) + (1-self.Y) * nn.LogSigmoid(-Xw))
          term_2 =  MultivariateNormal(torch.zeros(self.num_params_half), 
                                       (w_alphas + self.JITTER).diag()).log_prob(w_param).sum()
          term_3 =  self.prior.log_prob(torch.log(w_alphas**0.5)).sum()
        
          log_likelihood =  term_1 + term_2 + term_3
                
          return -log_likelihood # negative log_like
      else:
          Xw = torch.matmul(self.adjustedX.double(),w.double())
          term_1 =  torch.sum(self.Y * nn.LogSigmoid(Xw) + (1-self.Y) * nn.LogSigmoid(-Xw))
          term_2 =  self.prior.log_prob(w).sum()
        
          log_likelihood =  term_1 + term_2 
          return -log_likelihood # negative log_like
                
        
        
        
    
   


