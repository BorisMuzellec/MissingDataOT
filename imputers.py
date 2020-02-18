#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from geomloss import SamplesLoss

from utils import mean_impute, ns_bures, moments, MAE

import logging



class OTimputer():
    """
    'One parameter equals one imputed value' model (Algorithm 1. in the paper)


    Parameters
    ----------

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"


    """
    def __init__(self, 
                 eps=0.01, 
                 lr=1e-2, 
                 opt=torch.optim.RMSprop, 
                 niter = 2000, 
                 batchsize = 128, 
                 n_pairs = 1,
                 noise = 0.1, 
                 loss_func='sinkhorn', 
                 scaling = .9):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.loss_func = loss_func

        if loss_func == 'sinkhorn':
            self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, \
                          scaling=scaling, backend = "tensorized")

    def fit_transform(self, X, mask, verbose = True, report_interval=500,
                     X_true = None):

        """
        Imputes missing values using a batched OT loss

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor

        niter : int, default=2000

        batchsize : int, default=128

        lr : float, default=0.01

        tol : float, default=0.001

        weight_decay : float, default=1e-5

        verbose: bool, default=True

        order : str, default="random"
        Valid values: {"random" or "increasing"}.

        X_true: torch.DoubleTensor or None, default=None

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).


        """
        
        n, d = X.shape
        
        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        NAs = self.noise * torch.randn(mask.shape).double() + mean_impute(X, mask)
        NAs.requires_grad = True

        optimizer = self.opt([NAs], lr=self.lr)

        if verbose:
            logging.info(f"loss_func = {self.loss_func}, batchsize = {self.batchsize}, epsilon = {self.eps}")


        for i in range(self.niter):

            X_filled = (1 - mask) * X.detach() + mask * NAs
            loss = 0
            
            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)
    
                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
    
                if self.loss_func == 'sinkhorn':
                    loss = loss + self.sk(X1, X2)
    
                elif self.loss_func == 'bures':
                    m1, C1 = moments(X1)
                    m2, C2 = moments(X2)
                    loss = loss +  (((m1 - m2)**2).sum() + ns_bures(C1, C2))

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if  verbose  and (i % report_interval == 0):
                if X_true is not None:
                    logging.info('Iteration {}:\t {}\t Validation MAE {}'.format(i, 
                                 loss.item() / self.n_pairs,
                                 MAE((1 - mask) * X.detach() + \
                                     mask * NAs, X_true, mask).item()))
                else:
                    logging.info('Iteration {}:\t {}'.format(i,
                                 loss.item() / self.n_pairs))

        return (1 - mask) * X.detach() + mask * NAs



class RRimputer():
    """
    Round-Robin imputer with a batch sinkhorn loss

    Parameters
    ----------
    models: iterable
        iterable of torch.nn.Module. The j-th model is used to predict the j-th
        variable using all others.

    eps: float, default=0.01
        Sinkhorn regularization parameter.
        
    lr : float, default = 0.01
        Learning rate.

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.
        
    max_iter : int, default=10
        Maximum number of round-robin cycles for imputation.

    niter : int, default=15
        Number of gradient updates for each model within a cycle.

    batchsize : int, defatul=128
        Size of the batches on which the sinkhorn divergence is evaluated.

    n_pairs : int, default=10
        Number of batch pairs used per gradient update.

    tol : float, default = 0.001
        Tolerance threshold for the stopping criterion.

    weight_decay : float, default = 1e-5
        L2 regularization magnitude.

    order : str, default="random"
        Order in which the variables are imputed.
        Valid values: {"random" or "increasing"}.

    unsymmetrize: bool, default=True
        If True, sample one batch with no missing 
        data in each pair during training.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"

    """
    def __init__(self,
                 models, 
                 eps= 0.01, 
                 lr=1e-2, 
                 opt=torch.optim.Adam, 
                 max_iter=10,
                 niter=15, 
                 batchsize=128,
                 n_pairs=10, 
                 tol=1e-3, 
                 weight_decay=1e-5, 
                 order='random',
                 unsymmetrize=True, 
                 scaling=.9):

        self.models = models
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps,
                              scaling=scaling, backend="auto")
        self.lr = lr
        self.opt = opt
        self.max_iter = max_iter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.weight_decay=weight_decay
        self.order=order
        self.unsymmetrize = unsymmetrize

        self.is_fitted = False

    def fit_transform(self, X, mask,  verbose=True, 
                      report_interval=1, X_true=None):
        """
        Fits the imputer on a dataset with missing data, and returns the
        imputations.

        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose : bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """

        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        order_ = torch.argsort(mask.sum(0))

        optimizers = [self.opt(self.models[i].parameters(), \
                       lr=self.lr, weight_decay=self.weight_decay) 
                        for i in range(d)]

        X = (1 - mask) * X + mask * mean_impute(X, mask)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):

                j = order_[l].item()

                for k in range(self.niter):
                    
                    loss = 0
                    X_filled = X_filled.detach()
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()


                    for _ in range(self.n_pairs):
                        
                        idx1 = np.random.choice(n, self.batchsize, replace=False)
                        X1 = X_filled[idx1]

                        if self.unsymmetrize:
                            n_miss = (~mask[:, j].bool()).sum().item()
                            idx2 = np.random.choice(n_miss, self.batchsize, replace=False)
                            X2 = X_filled[~mask[:, j].bool(), :][idx2]

                        else:
                            idx2 = np.random.choice(n, self.batchsize, replace=False)
                            X2 = X_filled[idx2]

                        loss = loss + self.sk(X1, X2)

                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                ## Impute with last parameters
                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()


            if  verbose  and (i % report_interval == 0):
                if X_true is not None:
                    logging.info('Iteration {}:\t {}\t Validation MAE {}'.format(i,
                                 loss.item() / self.n_pairs, 
                                 MAE(X_filled, X_true, mask).item()))
                else:
                    logging.info('Iteration {}:\t {}'.format(i,
                                 loss.item() / self.n_pairs))

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')


        self.is_fitted = True

        return X_filled

    def transform(self, X, mask, verbose=True, report_interval=1,  X_true=None):
        """
        Impute missing values on new data. Assumes models have been previously 
        fitted on other data.
        
        Parameters
        ----------
        X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            Contains non-missing and missing data at the indices given by the
            "mask" argument. Missing values can be arbitrarily assigned 
            (e.g. with NaNs).

        mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
            mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.

        verbose: bool, default=True
            If True, output loss to log during iterations.
            
        report_interval : int, default=1
            Interval between loss reports (if verbose).

        X_true: torch.DoubleTensor or None, default=None
            Ground truth for the missing values. If provided, will output a 
            validation score during training. For debugging only.

        Returns
        -------
        X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
            Imputed missing data (plus unchanged non-missing data).

        """

        assert self.is_fitted, "The model has not been fitted yet."

        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        order_ = torch.argsort(mask.sum(0))

        X = (1 - mask) * X + mask * mean_impute(X, mask)
        X_filled = X.clone()

        for i in range(self.max_iter):

            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            for l in range(d):

                j = order_[l].item()

                with torch.no_grad():
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()


            if  verbose  and (i % report_interval == 0):
                if X_true is not None:
                    logging.info('Iteration {}:\t Validation MAE {}'.format(i,
                                 MAE(X_filled, X_true, mask).item()))

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        return X_filled
