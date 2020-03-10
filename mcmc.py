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

def K_lamda(lambd, x_values):
    """ Inputs: A length scale, lambd and a vector of x values, x
    Return: K matrix
    Description: The K matrix describes the covariance of the error
    """
    K = np.empty([len(x_values), len(x_values)])

    for i in range(len(x_values)):
        for j in range(len(x_values)):
            K[i][j] = np.exp(-lambd*pow((x_values[i]-x_values[j]), 2))
    return K

def K_zeta(zeta, x_values):
    """ Inputs: A length scale, exp(zeta) = lambda and a vector of x values, x
    Return: K matrix
    Description: The K matrix describes the covariance of the error
    """
    K = np.empty([len(x_values), len(x_values)])

    for i in range(len(x_values)):
        for j in range(len(x_values)):
            K[i][j] = np.exp(-np.exp(zeta)*pow((x_values[i]-x_values[j]), 2))
    return K

def K_zeta_sigma(zeta, sigma_delta, x_values):
    """ Inputs: A length scale, lambd and a vector of x values, x
    Return: K matrix
    Description: The K matrix describes the covariance of the error
    """
    K = np.empty([len(x_values), len(x_values)])

    for i in range(len(x_values)):
        for j in range(len(x_values)):
            K[i][j] = np.exp(-np.exp(zeta)*pow((x_values[i]-x_values[j]), 2))
    return np.multiply(pow(sigma_delta, 2), K)

def sigma(sigma, x_values):
    """ Inputs: A sigma, and a vector x with sensor position values.
    Returns: An isotropic covariance matrix with dimensions len(x), len(x)
    """ 
    SIGMA = pow(sigma, 2)*np.identity(len(x_values))
    return(SIGMA)
def error_mean(x_values):
    """ Inputs: A sigma, and a vector x with sensor position values.
    Returns: An isotropic covariance matrix with dimensions len(x), len(x)
    """ 
    ERROR = np.zeros(len(x_values))
    return(ERROR)
    
def likelihood(self, Y, U_VALUES, COV):
    """ Inputs:
        Returns:
        Description:
    """
    total_likelihood = 1
    
    if np.size(Y, 0)==1:
        likelihood = sp.multivariate_normal.pdf(Y, U_VALUES, COV)
        total_likelihood = likelihood
    else:
        
        for i in range(np.size(Y, 0)):
            likelihood = sp.multivariate_normal.pdf(Y[i], U_VALUES, COV)
            total_likelihood *= likelihood
    return(total_likelihood)
    
def likelihood_scaled(Y, U_VALUES, COV, MU):
    """ Inputs:
        Returns:
        Description:
    """
    total_likelihood = 1
    
    if np.size(Y, 0)==1:
        likelihood = sp.multivariate_normal.pdf(Y, np.multiply(MU, U_VALUES), COV)
        total_likelihood = likelihood
    else:
        
        for i in range(np.size(Y, 0)):
            likelihood = sp.multivariate_normal.pdf(Y[i], U_VALUES, COV)
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

def delta(K, NO_SAMPLES, x_values):
    """ Inputs:
        Returns:
        Description:
    """
    ERROR_MEAN = np.zeros(len(x_values))
    delta = sp.multivariate_normal.rvs(ERROR_MEAN, K, NO_SAMPLES)
    return(delta)

def epsilon(SIGMA, NO_SAMPLES, x_values):
    """ Inputs:
        Returns:
        Description:
    """
    ERROR_MEAN = np.zeros(len(x_values))
    epsilon = sp.multivariate_normal.rvs(ERROR_MEAN, SIGMA, NO_SAMPLES)
    return(epsilon)
    