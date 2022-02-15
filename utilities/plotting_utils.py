
def plot_ard_importances(results):
    if (len(results.keys()) == 0):
        raise Exception('Results object is empty. run the run_chains method first') 
    # ensure that the samplers return the samples and alphas seperately
    return

def predictive_performnce(results, x_test, y_test):
    if (len(results.keys()) == 0):
        raise Exception('Results object is empty. run the run_chains method first') 
    # mse, auc etc
    return

def plot_log_prob(results):
    if (len(results.keys()) == 0):
        raise Exception('Results object is empty. run the run_chains method first') 
    # likelihoods for different samplers
    return

def plot_ess(results):
    if (len(results.keys()) == 0):
        raise Exception('Results object is empty. run the run_chains method first') 
    # ess over time, grad val, func val
    return

def plot_r_hat(results):
    if (len(results.keys()) == 0):
        raise Exception('Results object is empty. run the run_chains method first') 
    return