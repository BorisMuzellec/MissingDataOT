#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

import itertools
import logging

from geomloss import SamplesLoss


from utils import mean_impute, ns_bures, moments, MAE


class OTimputer():
    """
    'One parameter equals one imputed value' model (Algorithm 1. in the paper)


    Parameters
    ----------
    eps : float, default=0.01

    opt : torch.optim.Optimizer, default=torch.optim.RMSprop

    loss_func : str, default="sinkhorn"
    Valid values: {"sinkhorn" or "bures"}.

    scaling : float, default=.9


    """
    def __init__(self, eps=0.01, opt=torch.optim.RMSprop,
                 loss_func='sinkhorn', scaling = .9):

        self.eps = eps
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, \
                              scaling=scaling, backend = "tensorized")
        self.opt = opt
        self.loss_func = loss_func

    def fit_transform(self, X, mask, niter = 2000, batchsize = 128, lr=1e-2,
                     noise = 0.1, verbose = True, report_interval=500,
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
        X_filled:


        """
        if batchsize > len(X) // 2:
            e = int(np.log2(len(X) // 2))
            batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {batchsize}.")

        NAs = noise * torch.randn(mask.shape).double() + mean_impute(X, mask)
        NAs.requires_grad = True

        optimizer = self.opt([NAs], lr=lr)

        if verbose:
            logging.info(f"loss_func = {self.loss_func}, batchsize = {batchsize}, epsilon = {self.eps}")


        for i in range(niter):

            X_filled = (1 - mask) * X.detach() + mask * NAs

            idx1 = np.random.choice(len(X), batchsize, replace=False)
            idx2 = np.random.choice(len(X), batchsize, replace=False)

            X1 = X_filled[idx1]
            X2 = X_filled[idx2]

            if self.loss_func == 'sinkhorn':
                loss = self.sk(X1, X2)

            elif self.loss_func == 'bures':
                m1, C1 = moments(X1)
                m2, C2 = moments(X2)
                loss =  (((m1 - m2)**2).sum() + ns_bures(C1, C2))

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                ### Catch numerical errors/overflows (should not happen)
                logging.info("Nan or inf loss")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if  verbose  and (i % report_interval == 0):
                if X_true is not None:
                    logging.info('Iteration {}:\t {}\t Validation MAE {}'.format(i, loss.item(),
                                 MAE((1 - mask) * X.detach() + mask * NAs, X_true, mask).item()))
                else:
                    logging.info('Iteration {}:\t {}'.format(i, loss.item()))

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

    opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
        Optimizer class to use for fitting.

    scaling: float, default=0.9
        Scaling parameter in Sinkhorn iterations
        c.f. geomloss' doc: "Allows you to specify the trade-off between
        speed (scaling < .4) and accuracy (scaling > .9)"

    """
    def __init__(self, models, eps= 0.01, opt=torch.optim.Adam, scaling=.9):

        self.models = models
        self.eps = eps
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps,
                              scaling=scaling, backend="auto")
        self.opt = opt

        self.is_fitted = False

    def fit_transform(self, X, mask, max_iter=10, niter=15, batchsize=128,
                      n_pairs=10, lr=1e-2, tol=1e-3, weight_decay =1e-5,
                      unsymmetrize=True, verbose=True, report_interval=1,
                      order='random', X_true=None):
        """
        Fits the imputer on a dataset with missing data, and returns the
        imputations.

        Parameters
        ----------
        X:

        mask:

        max_iter:

        niter:

        batchsize:

        n_pairs:

        lr:

        tol:

        weight_decay:

        verbose: bool, default=True

        order : str, default="random"
        Valid values: {"random" or "increasing"}.

        X_true:

        Returns
        -------
        X_filled:

        """

        n, d = X.shape
        normalized_tol = tol * torch.max(torch.abs(X[~mask.bool()]))

        if batchsize > n // 2:
            e = int(np.log2(n // 2))
            batchsize = 2**e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {batchsize}.")

        order_ = torch.argsort(mask.sum(0))

        optimizers = [self.opt(self.models[i].parameters(), \
                       lr=lr, weight_decay=weight_decay) for i in range(d)]

        X = (1 - mask) * X + mask * mean_impute(X, mask)
        X_fitted = X.clone()

        for i in range(max_iter):

            if order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_fitted.clone().detach()

            for l in range(d):

                j = order_[l].item()

                for k in range(niter):

                    loss = 0
                    X = X_fitted.clone().detach()
                    X_filled = X.clone()
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()


                    for _ in range(n_pairs):

                        if unsymmetrize:

                            n_miss = (~mask[:, j].bool()).sum().item()

                            idx1 = np.random.choice(n, batchsize, replace=False)
                            idx2 = np.random.choice(n_miss, batchsize, replace=False)

                            X1 = X_fitted[idx1]
                            X2 = X_fitted[~mask[:, j].bool(), :][idx2]

                        else:
                            idx1 = np.random.choice(n, batchsize, replace=False)
                            idx2 = np.random.choice(n, batchsize, replace=False)

                            X1 = X_fitted[idx1]
                            X2 = X_fitted[idx2]

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
                                 loss.item() / n_pairs, MAE(X_fitted, X_true, mask).item()))
                else:
                    logging.info('Iteration {}:\t {}'.format(i,
                                 loss.item() / n_pairs))

            if torch.norm(X_fitted - X_old, p=np.inf) < normalized_tol:
                break

        if i == (max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')


        self.is_fitted = True
        self.max_iter = max_iter
        self.order = order

        return X_filled

    def transform(self, X, mask, max_iter=None, order=None, tol=1e-3,
                  verbose=True, X_true=None, report_interval=1):
        """
        Parameters
        ----------
        X:

        mask:

        max_iter:

        lr:

        tol:

        verbose: bool, default=True

        order: str, default="random"
        Valid values: {"random" or "increasing"}.

        X_true:

        Returns
        -------
        X_filled:

        """

        assert self.is_fitted, "The model has not been fitted yet."

        ### Use the parameters used for fitting (unless stated otherwise)
        max_iter = self.mask_iter if max_iter is None else max_iter
        order = self.order if order is None else order

        n, d = X.shape
        normalized_tol = tol * torch.max(torch.abs(X[~mask.bool()]))


        order_ = torch.argsort(mask.sum(0))

        X = (1 - mask) * X + mask * mean_impute(X, mask)
        X_fitted = X.clone()

        for i in range(max_iter):

            if order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_fitted.clone().detach()

            for l in range(d):

                j = order_[l].item()

                with torch.no_grad():
                    X_fitted[mask[:, j].bool(), j] = self.models[j](X_fitted[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]).squeeze()


            if  verbose  and (i % report_interval == 0):
                if X_true is not None:
                    logging.info('Iteration {}:\t Validation MAE {}'.format(i,
                                 MAE(X_fitted, X_true, mask).item()))

            if torch.norm(X_fitted - X_old, p=np.inf) < normalized_tol:
                break

        if i == (max_iter - 1) and verbose:
            logging.info('Early stopping criterion not reached')

        return X_fitted
