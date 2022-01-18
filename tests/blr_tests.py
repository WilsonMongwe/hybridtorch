import unittest
from models.logistic_regression import LogisticRegression as BLR
import torch
import numpy as np
import torch.nn as nn


# test setup
X = torch.tensor([[0.1, 0.1], [0.2, 0.2]])
Y = torch.tensor([ 1, 0])
dim = 3
ard = True
model = BLR(X, Y, dim, ard)

weights = torch.tensor([0.1, 0.1, 1.0])

class TestStringMethods(unittest.TestCase):
    
    def test_blr_initialisation(self):
        self.assertTrue(np.array_equal(model.X.numpy(), X.numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(model.Y.numpy(), Y.numpy(), equal_nan=True))
        self.assertEqual(model.dimensions, dim)
        self.assertEqual(model.ard, ard)
        self.assertEqual(model.num_params, dim * 2)
        self.assertEqual(model.ALPHA, 1)
        self.assertEqual(model.JITTER, 1e-3)
    
    def test_blr_unflattern(self):
        w_expected, b_expected = torch.tensor([0.1  ,0.1]), torch.tensor([1])
        w_result, b_result = model.unflattern(weights)
        
        self.assertTrue(np.array_equal(w_result.numpy(), w_expected.numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(b_result.numpy(), b_expected.numpy(), equal_nan=True))
        
    def test_blr_predictions(self):
        result = model.predictions(X, weights)
        w, b = model.unflattern(weights)
        expected = torch.matmul(X,w) + b
        expected = nn.Sigmoid()(expected)
        self.assertTrue(np.allclose(expected.numpy(), result.numpy(), rtol = 1e-8, equal_nan=True,))

if __name__ == '__main__':
    unittest.main(verbosity=2)
   