# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:16:27 2020

@author: Ben Boys
"""
import numpy as np
import scipy.stats as sp
from scipy.special import gamma

def gamma_prior_pdf(zeta, alpha =1., beta=100.0):
    """ pdf of zeta rv with values of alpha and beta
    """
    NUM1 = pow(gamma(alpha), -1)
    NUM2 = pow(beta, alpha)
    EXP = zeta*(alpha) - beta*pow(np.e, zeta)
    
    return(NUM1*NUM2*pow(np.e, EXP))

def get_likelihood(Y, U_VALUES, COV):
    """ Inputs:
        Returns:
        Description:
    """
    total_likelihood = 1
    
    if np.size(Y, 0)==1:
        likelihood = sp.multivariate_normal.pdf(Y, U_VALUES, 1)
        total_likelihood = likelihood
        print('likelihood for realisation', total_likelihood)
    else:
        
        for i in range(np.size(Y, 0)):
            likelihood = sp.multivariate_normal.pdf(Y[i], U_VALUES, 1)
            total_likelihood *= likelihood
    return(total_likelihood)

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
