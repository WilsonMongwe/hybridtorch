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


def leapfrog_intergrator_sv(weights, momentum, step_size,
                         path_length, log_p, gradients, vol):
    
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * (dHdw)
    
    for t in range(path_length -1):
        weights = weights + (step_size * momentum)/vol
        dHdw = gradients(log_p(weights), weights)
        momentum = momentum - step_size * dHdw
    
    weights = weights + (step_size * momentum)/vol
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * dHdw

    return weights, -momentum



def magnetic_field_exp_and_factor(G, step_size):
    eig, eig_vec = torch.symeig(G, eigenvectors=True)
    eig = eig.numpy() 
    eig_vec = eig_vec.numpy() 
    
    positions = eig != 0.00000
    positions_new =  eig == 0.00000

    eig_new = eig[positions]
    lamb = torch.tensor(np.diag(eig_new))
    U_lamb = eig_vec[:, positions]
    U_0 = eig_vec[:,positions_new]

    lamb_exp =  torch.matrix_exp((lamb * step_size)).numpy()
    lamb_exp_minus =  torch.matrix_exp(lamb * step_size) - torch.eye(lamb.shape[0])
    lamb_exp_minus_new = np.matmul(torch.inverse(lamb).numpy(), lamb_exp_minus.numpy())
    
    term1 = np.matmul(np.matmul(U_lamb,lamb_exp), U_lamb.T)
    term2 = np.matmul(U_0,U_0.T)
    
    G_exp =  torch.tensor(term1 + term2)
    factor = np.matmul(np.matmul(U_lamb,lamb_exp_minus_new), U_lamb.T) 
    + step_size * np.matmul(U_0, U_0.T)
    
    return G_exp, torch.tensor(factor)


def magnetic_field_exp_and_factor_sv(G, step_size, volMat):
    newG = torch.matmul(G, volMat)
    eig, eig_vec = torch.symeig(newG, eigenvectors=True)
    eig = eig.numpy() 
    eig_vec = eig_vec.numpy() 
    positions = eig != 0.00000
    positions_new =  eig == 0.00000

    eig_new = eig[positions]
    lamb = torch.tensor(np.diag(eig_new))
    
    U_lamb = eig_vec[:, positions]
    U_0 = eig_vec[:,positions_new]

    lamb_exp =  torch.matrix_exp((lamb * step_size)).numpy()
    lamb_exp_minus =  torch.matrix_exp(lamb * step_size) - torch.eye(lamb.shape[0])
    lamb_exp_minus_new = np.matmul(torch.inverse(lamb).numpy(), lamb_exp_minus.numpy())
    
    term1 = np.matmul(np.matmul(U_lamb,lamb_exp), U_lamb.T)
    term2 = np.matmul(U_0,U_0.T)
    
    G_exp =  torch.tensor(term1 + term2)
    factor = np.matmul(np.matmul(U_lamb,lamb_exp_minus_new), U_lamb.T) + step_size * np.matmul(U_0, U_0.T)
    
    return G_exp, torch.tensor(factor)

def magnetic_leap_frog_intergrator(weights, momentum, step_size,
                                   path_length, log_p, gradients, G):

    G_exp, factor = magnetic_field_exp_and_factor(G, step_size)

    #step 1
    weights = weights    
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * dHdw
        
    for t in range(path_length-1): 
        # step 2
        weights = weights + torch.matmul(factor , momentum)
        momentum = torch.matmul(G_exp, momentum)
        #step 3
        weights = weights 
        dHdw = gradients(log_p(weights), weights)
        momentum = momentum - step_size * dHdw

    # step 2
    weights = weights + torch.matmul(factor , momentum)
    momentum = torch.matmul(G_exp, momentum)
    # step 3
    weights = weights
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum -0.5 * step_size * dHdw
    
    return weights, -momentum, -G


def magnetic_leap_frog_intergrator_sv(weights, momentum, step_size,
                                   path_length, log_p, gradients, G, variance):

    G_exp, factor = magnetic_field_exp_and_factor_sv(G, step_size, (1 / variance).diag())

    #step 1
    weights = weights    
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum - 0.5 * step_size * dHdw
        
    for t in range(path_length-1): 
        # step 2
        weights = weights + torch.matmul(factor , momentum) / variance
        momentum = torch.matmul(G_exp, momentum)
        #step 3
        weights = weights 
        dHdw = gradients(log_p(weights), weights)
        momentum = momentum - step_size * dHdw

    # step 2
    weights = weights + torch.matmul(factor , momentum) / variance
    momentum = torch.matmul(G_exp, momentum)
    # step 3
    weights = weights
    dHdw = gradients(log_p(weights), weights)
    momentum = momentum -0.5 * step_size * dHdw
    
    return weights, -momentum, -G


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

######################## ESS calc ##################################

def multiESS(X, b='sqroot', Noffsets=10, Nb=None):
    """
    Original source: https://github.com/lacerbi/multiESS
    Reference:
    Vats, D., Flegal, J. M., & Jones, G. L. "Multivariate Output Analysis
    for Markov chain Monte Carlo", arXiv preprint arXiv:1512.07713 (2015).
    """
    # MCMC samples and parameters
    n, p = X.shape

    if p > n:
        raise ValueError(
            "More dimensions than data points, cannot compute effective "
            "sample size.")

    # Input check for batch size B
    if isinstance(b, str):
        if b not in ['sqroot', 'cuberoot', 'less']:
            raise ValueError(
                "Unknown string for batch size. Allowed arguments are "
                "'sqroot', 'cuberoot' and 'lESS'.")
        if b != 'less' and Nb is not None:
            raise Warning(
                "Nonempty parameter NB will be ignored (NB is used "
                "only with 'lESS' batch size B).")
    else:
        if not 1. < b < (n / 2):
            raise ValueError(
                "The batch size B needs to be between 1 and N/2.")

    # Compute multiESS for the chain
    mESS = multiESS_chain(X, n, p, b, Noffsets, Nb)

    return mESS


def multiESS_chain(Xi, n, p, b, Noffsets, Nb):
    """
    Compute multiESS for a MCMC chain.
    """

    if b == 'sqroot':
        b = [int(np.floor(n ** (1. / 2)))]
    elif b == 'cuberoot':
        b = [int(np.floor(n ** (1. / 3)))]
    elif b == 'less':
        b_min = np.floor(n ** (1. / 4))
        b_max = max(np.floor(n / max(p, 20)), np.floor(np.sqrt(n)))
        if Nb is None:
            Nb = 20
        # Try NB log-spaced values of B from B_MIN to B_MAX
        b = set(map(int, np.round(np.exp(
            np.linspace(np.log(b_min), np.log(b_max), Nb)))))

    # Sample mean
    theta = np.mean(Xi, axis=0)
    # Determinant of sample covariance matrix
    if p == 1:
        detLambda = np.cov(Xi.T)
    else:
        detLambda = np.linalg.det(np.cov(Xi.T))

    # Compute mESS
    mESS_i = []
    for bi in b:
        mESS_i.append(multiESS_batch(Xi, n, p, theta, detLambda, bi, Noffsets))
    # Return lowest mESS
    mESS = np.min(mESS_i)

    return mESS


def multiESS_batch(Xi, n, p, theta, detLambda, b, Noffsets):
    """
    Compute multiESS for a given batch size B.
    """

    # Compute batch estimator for SIGMA
    a = int(np.floor(n / b))
    Sigma = np.zeros((p, p))
    offsets = np.sort(list(set(map(int, np.round(
        np.linspace(0, n - np.dot(a, b), Noffsets))))))

    for j in offsets:
        # Swapped a, b in reshape compared to the original code.
        Y = Xi[j + np.arange(a * b), :].reshape((a, b, p))
        Ybar = np.squeeze(np.mean(Y, axis=1))
        Z = Ybar - theta
        for i in range(a):
            if p == 1:
                Sigma += Z[i] ** 2
            else:
                Sigma += Z[i][np.newaxis, :].T * Z[i]

    Sigma = (Sigma * b) / (a - 1) / len(offsets)
    det = np.linalg.det(Sigma)
    ratio  = np.abs(detLambda / (det + 1e-200)) 
    exponent = 1.0/ p
    mESS = n*(ratio) ** exponent
    return mESS
