# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:16:27 2020

@author: Ben Boys
"""
import numpy as np
import scipy.stats as sp
from scipy.special import gamma

def gamma_prior_pdf(x, alpha =1., beta=100.0):
    """ pdf of zeta rv with values of alpha and beta
    """
    NUM1 = pow(gamma(alpha), -1)
    NUM2 = pow(beta, alpha)
    EXP = x*(alpha) - beta*pow(np.e, x)
    
    return(NUM1*NUM2*pow(np.e, EXP))


def get_fast_likelihood(Y, U_VALUES):
    # Since cov = identity, then
    l2 = np.dot(Y, U_VALUES)
    
    nll = 1./2 * l2
    print(nll)
    likelihood = np.exp(-1.*nll)
    print(likelihood)
    return likelihood

def beta_likelihood(damage_data, model_sample):
    mode = model_sample
    alpha = 2
    beta = 1/mode
    log_likelihood = 0
    for i in range(len(damage_data)):
        likelihood = sp.beta.pdf(damage_data, alpha, beta, loc = model_sample)
        log_likelihood += np.log(likelihood)
    total_likelihood = np.exp(log_likelihood)
    return total_likelihood
