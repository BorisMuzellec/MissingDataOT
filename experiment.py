#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

from geomloss import SamplesLoss

import ot

import os
import pickle as pkl
import copy

from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils import *
from softimpute import softimpute, cv_softimpute
from data_loaders import dataset_loader
from imputers import OTimputer, RRimputer

import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default=None,
                    help='filename for the results')
parser.add_argument('--out_data', type=str, default=None,
                    help='filename for the data')
parser.add_argument('--out_dir', type=str, default='exps',
                    help='directory name for results')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--decay', type=float, default=1e-5,
                    help='weight decay (round robin)')
parser.add_argument('--scaling', type=float, default=.9,
                    help='sinkhorn scaling parameter (speed/precision tradeoff)')
parser.add_argument('-b', '--batchsize', type=int, default=128,
                    help='batchsize(s) for the experiments')
parser.add_argument('--niter', type=int, default=3000,
                    help='number of GD iterations')
parser.add_argument('--max_iter', type=int, default=15,
                    help='maximum number of cycles (round robin)')
parser.add_argument('--rr_niter', type=int, default=15,
                    help='number of GD iterations (round robin)')
parser.add_argument('--n_pairs', type=int, default=10,
                    help='number of pairs batches to sample (round robin)')
parser.add_argument('-e', '--epsilon', type=float, default=None,
                    help='Sinkhorn regularization parameter. '
                         'Automatically select using median distance by default')
parser.add_argument('--quantile', type=float, default=.5,
                    help='distance quantile to select epsilon')
parser.add_argument('-qm', '--quantile_multiplier', type=float, default=0.05,
                    help='distance quantile x multiplier =  epsilon')
parser.add_argument('--nexp', type=int, default=1,
                    help='number of experiences per parameter setting')
parser.add_argument('--dataset', type=str, default="iris",
                    help='dataset on which to run the experiments')
parser.add_argument('--p', type=float, default=0.3, help='Proportion of imps')
parser.add_argument('--MAR', action='store_true')
parser.add_argument('--p_obs', type=float, default=0.3,
                    help='Proportion of variables that are fully observed (MAR & MNAR model)')
parser.add_argument('--MNAR_log', action='store_true')
parser.add_argument('--MNAR_quant', action='store_true')
parser.add_argument('--q_mnar', type=float, default=0.75,
                    help='quantile that will have imps (MNAR quantiles model)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--report_interval', type=int, default=500)

args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

if __name__ == "__main__":

    OTLIM = 5000

    dataset = args.dataset

    ground_truth = scale(dataset_loader(dataset))
    X_true = torch.tensor(ground_truth)

    METHODS = ["OT", "ice", "mean", "softimpute", "lin_rr", "mlp_rr"]

    ot_scores = {}
    ice_scores = {}
    mean_scores = {}
    softimpute_scores = {}
    lin_rr_scores = {}
    mlp_rr_scores = {}

    score_dicts = [ot_scores, ice_scores, mean_scores, softimpute_scores,
                   lin_rr_scores, mlp_rr_scores]

    for dic in score_dicts:
        for metric in ['MAE', 'RMSE', 'OT']:
            dic[metric] = []

    p = args.p

    data = {"p": p, "ground_truth": ground_truth, "mask": [], "M": [],
            "epsilon": [], "imp": {}, "params": vars(args)}

    for meth in METHODS:
        data["imp"][meth] = []

    batchsize = args.batchsize

    for n in range(args.nexp):

        ### Each entry from the second axis has a probability p of being NA

        if args.MAR:
            logging.info("MAR")
            mask = MAR_mask(X_true, p, args.p_obs).double()
        elif args.MNAR_log:
            logging.info("Logistic MNAR")
            mask = MNAR_mask_logistic(X_true, p, args.p_obs).double()
        elif args.MNAR_quant:
            logging.info("Quantile MNAR")
            mask = MNAR_mask_quantiles(X_true, p, args.q_mnar, 1-args.p_obs,
                                       cut='both', MCAR=False).double()
        else:
            mask = (torch.rand(ground_truth.shape) < p).double()

        X_nas = X_true.clone()
        X_nas[mask.bool()] = np.nan

        M = mask.sum(1) > 0
        nimp = M.sum().item()

        data["mask"].append(mask.detach().cpu().numpy())
        data["M"].append(M.detach().cpu().numpy())

        ice_mean = IterativeImputer(random_state=0, max_iter=50)
        data_nas = X_nas.cpu().numpy()
        ice_mean.fit(X_nas.cpu().numpy())

        ice_imp = torch.tensor(ice_mean.transform(data_nas))
        mean_imp = (1 - mask) * X_true + mask * nanmean(X_nas)

        data["imp"]["ice"].append(ice_imp.cpu().numpy())
        data["imp"]["mean"].append(mean_imp.cpu().numpy())

        mean_scores['MAE'].append(MAE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['RMSE'].append(RMSE(mean_imp, X_true, mask).cpu().numpy())
        if nimp < OTLIM:
            dists = ((mean_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            mean_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                             np.ones(nimp) / nimp,
                                             dists.cpu().numpy()))

            logging.info(f'mean imputation:\t '
                         f'MAE: {mean_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {mean_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {mean_scores["OT"][-1]:.4f}')
        else:
            logging.info(f'mean imputation:\t '
                         f'MAE: {mean_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {mean_scores["RMSE"][-1]:.4f}')

        ice_scores['MAE'].append(MAE(ice_imp, X_true, mask).cpu().numpy())
        ice_scores['RMSE'].append(RMSE(ice_imp, X_true, mask).cpu().numpy())
        if nimp < OTLIM:
            dists = ((ice_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            ice_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                            np.ones(nimp) / nimp,
                                            dists.cpu().numpy()))
            logging.info(f'ice imputation:\t'
                         f'MAE: {ice_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {ice_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {ice_scores["OT"][-1]:.4f}')
        else:
            logging.info(f'ice imputation:\t'
                         f'MAE: {ice_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {ice_scores["RMSE"][-1]:.4f}')

        cv_error, grid_lambda = cv_softimpute(data_nas, grid_len=15)
        lbda = grid_lambda[np.argmin(cv_error)]

        softimp = softimpute((data_nas), lbda)[1]

        data["imp"]["softimpute"].append(softimp)
        softimp = torch.tensor(softimp)
        softimpute_scores['MAE'].append(
            MAE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['RMSE'].append(
            RMSE(softimp, X_true, mask).cpu().numpy())
        if nimp < OTLIM:
            dists = ((softimp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            softimpute_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                                   np.ones(nimp) / nimp,
                                                   dists.cpu().numpy()))
            logging.info(f'softimpute:\t'
                         f'MAE: {softimpute_scores["MAE"][-1]:.4f}\t'
                         f'RMSE: {softimpute_scores["RMSE"][-1]:.4f}\t'
                         f'OT: {softimpute_scores["OT"][-1]:.4f}')
        else:
            logging.info(f'softimpute:\t'
                         f'MAE: {softimpute_scores["MAE"][-1]:.4f}\t '
                         f'RMSE: {softimpute_scores["RMSE"][-1]:.4f}')

        ### Automatic epsilon

        if args.quantile is not None:
            epsilon = pick_epsilon(X_nas, args.quantile, args.quantile_multiplier)
            logging.info(f"epsilon: {epsilon:.4f} "
                         f"({100 * args.quantile}th percentile times "
                         f"{args.quantile_multiplier})")

        else:
            epsilon = args.epsilon
            logging.info(f"epsilon: {epsilon:.4f} (fixed)")

        data["epsilon"].append(epsilon)

        logging.info("Sinkhorn Imputation")

        sk_imputer = OTimputer(eps=epsilon, niter=args.niter, batchsize=batchsize, lr=args.lr)

        sk_imp, _, _ = sk_imputer.fit_transform(X_nas.clone(), report_interval=args.report_interval,
                                     verbose=True, X_true=X_true)
        sk_imp = sk_imp.detach()

        ot_scores['MAE'].append(MAE(sk_imp, X_true, mask).item())
        ot_scores['RMSE'].append(RMSE(sk_imp, X_true, mask).item())
        if nimp < OTLIM:
            dists = ((sk_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            ot_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                           np.ones(nimp) / nimp, \
                                           dists.cpu().numpy()))

            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {ot_scores['OT'][-1]:.4f}")
        else:
            logging.info(f"Sinkhorn imputation:\t "
                         f"MAE: {ot_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {ot_scores['RMSE'][-1]:.4f}")

        data["imp"]["OT"].append(sk_imp[mask.bool()].detach().cpu().numpy())

        logging.info("Linear Round Robin Imputation")

        n, d = X_true.shape

        models = {}

        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = torch.nn.Linear(d - 1, 1).to(device)

        linear_rr_imputer = RRimputer(models, max_iter=args.max_iter,
                                      niter=args.rr_niter,
                                      n_pairs=args.n_pairs,
                                      batchsize=batchsize,
                                      lr=args.lr,
                                      weight_decay=args.decay,
                                      order="random",
                                      eps=epsilon,
                                      opt=torch.optim.Adam,
                                      scaling=args.scaling)

        lin_imp, _, _ = linear_rr_imputer.fit_transform(X_nas.clone(), report_interval=1, verbose=True, X_true=X_true)
        lin_imp = lin_imp.detach()

        lin_rr_scores['MAE'].append(MAE(lin_imp, X_true, mask).item())
        lin_rr_scores['RMSE'].append(RMSE(lin_imp, X_true, mask).item())
        if nimp < OTLIM:
            dists = ((lin_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            lin_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                               np.ones(nimp) / nimp,
                                               dists.cpu().numpy()))
            logging.info(f"Linear RR imputation:\t"
                         f"MAE: {lin_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {lin_rr_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {lin_rr_scores['OT'][-1]:.4f}")
        else:
            logging.info(f"Linear RR imputation:\t"
                         f"MAE: {lin_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {lin_rr_scores['RMSE'][-1]:.4f}")

        data["imp"]["lin_rr"].append(lin_imp[mask.bool()].detach().cpu().numpy())

        logging.info("MLP Round Robin Imputation")

        n, d = X_true.shape
        d_ = d - 1

        models = {}

        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = nn.Sequential(nn.Linear(d_, 2 * d_),
                                      nn.ReLU(),
                                      nn.Linear(2 * d_, d_),
                                      nn.ReLU(),
                                      nn.Linear(d_, 1)
                                      ).to(device)

        mlp_rr_imputer = RRimputer(models,
                                   max_iter=args.max_iter,
                                   niter=args.rr_niter,
                                   n_pairs=args.n_pairs,
                                   batchsize=batchsize,
                                   lr=args.lr,
                                   weight_decay=args.decay,
                                   order="random",
                                   eps=epsilon,
                                   opt=torch.optim.Adam,
                                   scaling=args.scaling)

        mlp_imp, _, _ = mlp_rr_imputer.fit_transform(X_nas.clone(), report_interval=1, verbose=True, X_true=X_true)
        mlp_imp = mlp_imp.detach()

        mlp_rr_scores['MAE'].append(MAE(mlp_imp, X_true, mask).item())
        mlp_rr_scores['RMSE'].append(RMSE(mlp_imp, X_true, mask).item())
        if nimp < OTLIM:
            dists = ((mlp_imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
            mlp_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp,
                                               np.ones(nimp) / nimp,
                                               dists.cpu().numpy()))
            logging.info(f"MLP RR imputation:\t"
                         f"MAE: {mlp_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {mlp_rr_scores['RMSE'][-1]:.4f}\t"
                         f"OT: {mlp_rr_scores['OT'][-1]:.4f}")
        else:
            logging.info(f"MLP RR imputation:\t"
                         f"MAE: {mlp_rr_scores['MAE'][-1]:.4f}\t"
                         f"RMSE: {mlp_rr_scores['RMSE'][-1]:.4f}")

        data["imp"]["mlp_rr"].append(mlp_imp[mask.bool()].detach().cpu().numpy())

    scores = {}
    scores['OT'] = ot_scores
    scores['ice'] = ice_scores
    scores['mean'] = mean_scores
    scores['softimpute'] = softimpute_scores
    scores['lin_rr'] = lin_rr_scores
    scores['mlp_rr'] = mlp_rr_scores

    if args.out_path is None:
        score_file = "_".join([dataset, "scores.pkl"])
    else:
        score_file = args.out_path

    pkl.dump(scores, open(os.path.join(args.out_dir, score_file), 'wb'))

    if args.out_data is None:
        data_file = "_".join([dataset, "data.pkl"])
    else:
        data_file = args.out_data

    pkl.dump(data, open(os.path.join(args.out_dir, data_file), 'wb'))
