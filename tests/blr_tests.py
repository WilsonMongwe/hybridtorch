import unittest
from models.logistic_regression import LogisticRegression as BLR
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal


# test setup
X = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
adjustedX = torch.cat((X, torch.ones(2,1)), 1)
Y = torch.tensor([ 1, 0])
dim = 3
ard = True
model = BLR(X, Y, dim, ard)
model_2 = BLR(X, Y, dim, False)

weights = torch.tensor([0.1, 0.1, 1.0])
prior = Normal(loc = 0, scale = model.ALPHA , validate_args= False)

class TestStringMethods(unittest.TestCase):
    
    def test_blr_initialisation(self):
        self.assertTrue(np.array_equal(model.X.detach().numpy(), X.detach().numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(model.adjustedX.detach().numpy(), adjustedX.detach().numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(model.Y.detach().numpy(), Y.detach().numpy(), equal_nan=True))
        self.assertEqual(model.dimensions, dim)
        self.assertEqual(model.ard, ard)
        self.assertEqual(model.num_params, dim * 2)
        self.assertEqual(model.ALPHA, 1)
        self.assertEqual(model.JITTER, 1e-3)
    
    def test_blr_unflattern(self):
        w_expected, b_expected = torch.tensor([0.1  ,0.1]), torch.tensor([1])
        w_result, b_result = model.unflattern(weights)
        
        self.assertTrue(np.array_equal(w_result.detach().numpy(), w_expected.detach().numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(b_result.detach().numpy(), b_expected.detach().numpy(), equal_nan=True))
        
    def test_blr_predictions(self):
        result = model.predictions(X, weights)
        w, b = model.unflattern(weights)
        expected = torch.matmul(X,w) + b
        expected = nn.Sigmoid()(expected)
        self.assertTrue(np.allclose(expected.detach().numpy(), result.detach().numpy(), rtol = 1e-8, equal_nan=True,))
        
    def test_blr_log_prob_no_ard(self):
        Xw = torch.matmul(adjustedX, weights)
        term_1 =  torch.sum(Y * nn.LogSigmoid()(Xw) + (1-Y) * nn.LogSigmoid()(-Xw))
        term_2 =  prior.log_prob(weights).sum()
        expected = -(term_1 + term_2)
        result = model_2.log_prob(weights)
        self.assertEqual(result.detach().numpy(), expected.numpy())
        

if __name__ == '__main__':
    unittest.main(verbosity=2)
   