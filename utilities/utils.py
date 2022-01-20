import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def adaptation_params(step_size):
    Hbar_old = 0
    eps_bar = 1
    mu = np.log(10 * step_size)
    return Hbar_old, eps_bar, mu


def metropolis_acceptance_step(u, old_hamiltonian, new_hamiltonian,i):
    delta = new_hamiltonian - old_hamiltonian 
    alpha = torch.min(torch.tensor(1.0), torch.exp(-delta)) 
    if u < alpha:
        return True, alpha
    else:
        return False, alpha
    
def adaptation(m, mu, Hbar, acceptance_rate, eps_bar, desired_accept_rate):
    m = m + 1
    t0 = 10.0
    gamma = 0.05
    kappa = 0.75

    if np.isnan(acceptance_rate):
        acceptance_rate = 0
    
    Hbar_m = (1-(1/(m+t0))) * Hbar + (1/(m+t0))*(desired_accept_rate - acceptance_rate)
    
    log_eps_m = mu - (m**0.5)/gamma * Hbar_m
    log_eps_m_bar = m**(-kappa) * log_eps_m + (1-m**(-kappa)) * np.log(eps_bar)
    
    step_size = float(np.exp(torch.FloatTensor([log_eps_m])))
    
    eps_bar = float(np.exp(log_eps_m_bar))
    
    return step_size, eps_bar, Hbar_m 

def leapfrog_intergrator(weights, momentum, step_size,
                         path_length, log_p, gradients):
    
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * (dHdw)
    
    for t in range(path_length -1):
        weights = weights + (step_size * momentum)
        dHdw = gradients(log_p(weights), weights)
        momentum = momentum - step_size * dHdw
    
    weights = weights + (step_size * momentum)
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * dHdw

    return weights, -momentum

#################### Process datasets ###########################

def audit_outcomes_dataset_data():
    dataset = pd.read_csv("audit_outcomes_dataset.csv", header= 0) 
    
    dataset = dataset.drop(columns = ["demarcation.code"])
    dataset.head()
    dataset[["opinion.code"]] = (np.logical_and((dataset[["opinion.code"]] !=  "unqualified").values,
                             (dataset[["opinion.code"]] !=  "unqualified_emphasis_of_matter").values) *1)
    
     #Split the data into training and test set time wise
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values
    
    x_train = X[0:1404]
    y_train = y[0:1404]
    
    x_test = X[1404: X.shape[0]]
    y_test = y[1404: y.shape[0]]
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform (x_test)
    return torch.tensor(x_train), torch.tensor(y_train), torch.tensor(x_test), torch.tensor(y_test)

