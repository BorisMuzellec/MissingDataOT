#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


##### Mean Imputation #####

# def mean_impute(X, mask):
#     """
#     mask[i,j] should be 1. (or True) iff X[i,j] is missing
#     """
#     m = 1 - mask
#     return (X * m).sum(0) / m.sum(0)


def nanmean(v, *args, **kwargs):
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

#### Quantile ######

def quantile(X, q, dim=None):
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Accuracy Metrics ####

def MAE(X, X_true, mask):
    return torch.abs(X[mask.bool()] - X_true[mask.bool()]).sum() / mask.sum()

def RMSE(X, X_true, mask):
    return (((X[mask.bool()] - X_true[mask.bool()])**2).sum() / mask.sum()).sqrt()

##################### MISSING DATA MECHANISMS #############################

##### MAR ######

def MAR_mask(X_true, p, p_obs):

    n, d = X_true.shape

    mask = torch.zeros(n, d).bool()

    d_obs = max(int(p_obs * d), 1) ## number of variables that have no imps
    d_na = d - d_obs ## number of variables that have imps

    ### Sample variables that will all be observed:
    idxs_obs = np.random.choice(d, d_obs, replace=False) ### select at least one variable with no imps
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    coeffs = torch.randn((d_obs, d_na))
    intercepts = torch.randn(d_na)

    ps = torch.sigmoid(X_true[:, idxs_obs].mm(coeffs) + intercepts)
    ### Rescale to have a desired amount of missing values
    ps /= (ps.mean(0) / p)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


##### MNAR ######

def MNAR_mask_logistic(X_true, p, p_params):

    ### Same mechanism as logistic MAR,
    ### but the the variables of the logistic regression can now also be missing

    n, d = X_true.shape

    mask = torch.zeros(n, d).bool()

    d_params = max(int(p_params * d), 1) ## number of variables used as inputs
    d_na = d - d_params ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) ### select at least one variable
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    coeffs = torch.randn((d_params, d_na))
    intercepts = torch.randn(d_na)

    ps = torch.sigmoid(X_true[:, idxs_params].mm(coeffs) + intercepts)
    ### Rescale to have a desired amount of missing values
    ps /= (ps.mean(0) / p)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## Now, mask some values used in the logistic model at random
    ## This makes the missingness of other variables potentially dependent on masked values
    mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def MNAR_mask_quantiles(X_true, p, q, p_params, cut='both', MCAR = False):
    """
    cut can take values 'upper', 'lower' and 'both'. Determines which ends have imps
    """
    n, d = X_true.shape

    mask = torch.zeros(n, d).bool()

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR imps

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with imps

    ### check if values are greater/smaller that corresponding Nas

    if cut == 'upper':
        quants = quantile(X_true[:, idxs_na], q, dim=0)
        m = X_true[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X_true[:, idxs_na], 1-q, dim=0)
        m = X_true[:, idxs_na] <= quants
    elif cut == 'both':
        l_quants = quantile(X_true[:, idxs_na], 1-q, dim=0)
        u_quants = quantile(X_true[:, idxs_na], q, dim=0)
        m = (X_true[:, idxs_na] <= l_quants) | (X_true[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m


    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)


    return mask
