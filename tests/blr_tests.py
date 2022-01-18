import unittest
from models.logistic_regression import LogisticRegression as BLR
import torch
import numpy as np


X = torch.tensor([[0.1  ,0.1], [0.2 ,0.2]])
Y = torch.tensor([ 1, 0])
dim = 3
ard = True
model = BLR(X, Y, dim, ard)

class TestStringMethods(unittest.TestCase):
    
    def test_blr_initialisation(self):
        self.assertTrue(np.array_equal(model.X.numpy() , X.numpy(), equal_nan=True))
        self.assertTrue(np.array_equal(model.Y.numpy() , Y.numpy(), equal_nan=True))
        self.assertEqual(model.dimensions , dim)
        self.assertEqual(model.ard , ard)
        self.assertEqual(model.num_params , dim * 2)
        self.assertEqual(model.num_params_half , dim)

if __name__ == '__main__':
    unittest.main()
   