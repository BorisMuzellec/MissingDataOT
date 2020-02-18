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

from utils import moments, ns_bures, MAE, RMSE, quantile, MAR_mask, MNAR_mask_logistic, MNAR_mask_quantiles, mean_impute
from softimpute import softimpute, cv_softimpute
from data_loaders import dataset_loader
from imputers import OTimputer, RRimputer


import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type = str, default = None, help='filename for the results')
parser.add_argument('--out_data', type = str, default = None, help='filename for the data')
parser.add_argument('--out_dir', type = str, default = 'exps', help='directory name for results')
parser.add_argument('--lr', type=float, default = 1e-2, help='learning rate')
parser.add_argument('--decay', type=float, default = 1e-5, help='weight decay (round robin)')
parser.add_argument('--scaling', type=float, default = .9, help='sinkhorn scaling parameter (speed/precision tradeoff)')
parser.add_argument('-b', '--batchsize', type=int, default = 128, help='batchsize(s) for the experiments')
parser.add_argument('--niter', type=int, default=3000, help='number of GD iterations')
parser.add_argument('--max_iter', type=int, default=15, help='maximum number of cycles (round robin)')
parser.add_argument('--rr_niter', type=int, default=15, help='number of GD iterations (round robin)')
parser.add_argument('--n_pairs', type=int, default=10, help='number of pairs batches to sample (round robin)')
parser.add_argument('-g', '--gamma', type=float, default = None, 
                    help='Sinkhorn regularization parameter. Automatically select using median distance by default')
parser.add_argument('--quantile', type=float, default=.5, help='distance quantile to select gamma')
parser.add_argument('-qm', '--quantile_multiplier', type=float, default=0.05, help='distance quantile x multiplier =  gamma')
parser.add_argument('--nexp', type=int, default=30, help='number of experiences per parameter setting')
parser.add_argument('--dataset', type=str, default = "boston", help='dataset on which to run the experiments')
parser.add_argument('--p', type=float, default = 0.3, help='Proportion of NAs')
parser.add_argument('--MAR', action='store_true')
parser.add_argument('--p_obs', type=float, default = 0.3, 
                    help='Proportion of variables that are fully observed (MAR & MNAR model)')
parser.add_argument('--MNAR_log', action='store_true')
parser.add_argument('--MNAR_quant', action='store_true')
parser.add_argument('--q_mnar', type=float, default=0.25, 
                    help='quantile that will have NAs (MNAR quantiles model)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--report_interval', type=int, default = 500)

args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)



if __name__ == "__main__":

    
    OTLIM = 5000
    

    dataset = args.dataset

    ground_truth = scale(dataset_loader(dataset))
    X_true = torch.tensor(ground_truth)
    
    
    ot_scores = {}
    scikit_scores = {}  
    bures_scores = {}
    mean_scores = {}
    softimpute_scores = {}
    lin_rr_scores = {}
    mlp_rr_scores = {}
    
    score_dicts = [ot_scores, scikit_scores, bures_scores,\
                   mean_scores, softimpute_scores, 
                   lin_rr_scores, mlp_rr_scores]
    
    
    for dic in score_dicts:
        for metric in ['MAE', 'RMSE', 'OT', 'bures']:
            dic[metric] = []


    p = args.p

    data = {}
    data["p"] = p
    data["ground_truth"] = ground_truth
    data["mask"] = [] 
    data["M"] = [] 
    data["gamma"] = []
    
    data["imp"] = {}
    data["imp"]["OT"] = []
    data["imp"]["bures"] = []
    data["imp"]["scikit"] = []
    data["imp"]["mean"] = []
    data["imp"]["softimpute"] = []
    
    data["imp"]["lin_rr"] = []
    data["imp"]["mlp_rr"] = []
    
    data["params"] = vars(args)
    
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
            mask = MNAR_mask_quantiles(X_true, p, args.q_mnar, args.p_obs,
                                       cut='both', MCAR=False).double()
        else:
            mask = (torch.rand(ground_truth.shape) < p).double()
            
        M = mask.sum(1) > 0
        nimp = M.sum().item()
        mean_truth, cov_truth = moments(X_true[M])

        data["mask"].append(mask.detach().cpu().numpy())
        data["M"].append(M.detach().cpu().numpy())

        imp_mean = IterativeImputer(random_state=0, max_iter = 50)
        data_nas = copy.deepcopy(ground_truth)
        data_nas[data["mask"][-1] == 1] = np.nan
        imp_mean.fit(data_nas)
        
        sk_imp = torch.tensor(imp_mean.transform(data_nas))
        mean_imp = (1 - mask) * X_true + mask * mean_impute(X_true, mask)
        
        mean_sk, cov_sk = moments(sk_imp[M])
        mean_mean, cov_mean = moments(mean_imp[M])
        
        data["imp"]["scikit"].append(sk_imp.cpu().numpy())
        data["imp"]["mean"].append(mean_imp.cpu().numpy())
        
        scikit_scores['MAE'].append(MAE(sk_imp, X_true, mask).cpu().numpy())
        scikit_scores['RMSE'].append(RMSE(sk_imp, X_true, mask).cpu().numpy())
        scikit_scores['bures'].append([((mean_sk - mean_truth)**2).sum().item(), ns_bures(cov_sk, cov_truth).item()])
        if nimp < OTLIM:
            scikit_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                         ((sk_imp[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.)
            logging.info(f'scikit imputation :  MAE = {scikit_scores["MAE"][-1]}, OT = {scikit_scores["OT"][-1]}')
        else:
            logging.info(f'scikit imputation :  MAE = {scikit_scores["MAE"][-1]}')

        mean_scores['MAE'].append(MAE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['RMSE'].append(RMSE(mean_imp, X_true, mask).cpu().numpy())
        mean_scores['bures'].append([((mean_mean - mean_truth)**2).sum().item(), ns_bures(cov_mean, cov_truth).item()])
        if nimp < OTLIM:
            mean_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                       ((mean_imp[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.) 
            logging.info(f'mean imputation :  MAE = {mean_scores["MAE"][-1]}, OT = {mean_scores["OT"][-1]}')
        else:
            logging.info(f'mean imputation :  MAE = {mean_scores["MAE"][-1]}')
            

        cv_error, grid_lambda = cv_softimpute(data_nas, grid_len=15)
        lbda = grid_lambda[np.argmin(cv_error)]
        
        softimp = softimpute((data_nas), lbda)[1]

        
        data["imp"]["softimpute"].append(softimp)
        softimp = torch.tensor(softimp)
        mean_soft, cov_soft = moments(softimp[M])
        softimpute_scores['MAE'].append(MAE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['RMSE'].append(RMSE(softimp, X_true, mask).cpu().numpy())
        softimpute_scores['bures'].append([((mean_soft - mean_truth)**2).sum().item(), ns_bures(cov_soft, cov_truth).item()])
        if nimp < OTLIM:
            softimpute_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                   ((softimp[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.) 
            logging.info(f'softimpute :  MAE = {softimpute_scores["MAE"][-1]}, OT = {softimpute_scores["OT"][-1]}')
        else:
            logging.info(f'softimpute :  MAE = {softimpute_scores["MAE"][-1]}')


        ### Automatic gamma
        
        
        if args.quantile is not None:
        
            NAs = 0.1 * torch.randn(mask.shape).double() + mean_impute(X_true, mask)
            X_ = (1 - mask) * X_true + mask * NAs
            idx = np.random.choice(len(X_), min(2000, len(X_)), replace=False)
            X = X_[idx]
            dists = ((X[:, None] - X)**2).sum(2).flatten() / 2.
            dists = dists[dists > 0]
            gamma = quantile(dists, args.quantile, 0).item() * args.quantile_multiplier
            logging.info(f"epsilon = {gamma} ({100 * args.quantile}th percentile times {args.quantile_multiplier})")          
        
        else:
            
            gamma = args.gamma
            logging.info(f"epsilon = {gamma} (fixed)")
            

        data["gamma"].append(gamma)
        
        logging.info("Sinkhorn Imputation")
        
        sk = SamplesLoss("sinkhorn", p=2, blur=gamma, scaling=.9, backend = "tensorized")
        
        sk_imputer = OTimputer(eps=gamma, loss_func='sinkhorn', niter = args.niter, 
                               batchsize=batchsize, lr=args.lr)
        
        X = (1 - mask) * X_true.clone() + mask * mean_impute(X_true, mask)
        
        X = sk_imputer.fit_transform(X.clone(), mask, report_interval=args.report_interval,
                           verbose=True, X_true=X_true).detach()
        
        mean_ot, cov_ot = moments(X[M])

        ot_scores['MAE'].append(MAE(X, X_true, mask).item())
        ot_scores['RMSE'].append(RMSE(X, X_true, mask).item())
        ot_scores['bures'].append([((mean_ot - mean_truth)**2).sum().item(), ns_bures(cov_ot, cov_truth).item()])
        if nimp < OTLIM:
            ot_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                     ((X[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.)
            logging.info(f"batchsize = {batchsize}, MAE = {ot_scores['MAE'][-1]}, OT = {ot_scores['OT'][-1]}")
        else:
            logging.info(f"batchsize = {batchsize}, MAE = {ot_scores['MAE'][-1]}")
            
        data["imp"]["OT"].append(NAs[mask.bool()].detach().cpu().numpy())
                

        logging.info("Bures Imputation")
        
        bures_imputer = OTimputer(eps=gamma, niter = 2 * args.niter, \
                              batchsize = batchsize, lr= 0.5 * args.lr, 
                              opt = torch.optim.Adam, loss_func='bures')
        
        X = (1 - mask) * X_true.clone() + mask * mean_impute(X_true, mask)
        
        X = bures_imputer.fit_transform(X.clone(), mask, 
                              report_interval=args.report_interval, 
                              verbose=True, X_true=X_true).detach()

        mean_bures, cov_bures = moments(X[M])
        
        bures_scores['MAE'].append(MAE(X, X_true, mask).item())
        bures_scores['RMSE'].append(RMSE(X, X_true, mask).item())
        bures_scores['bures'].append([((mean_bures - mean_truth)**2).sum().item(), ns_bures(cov_bures, cov_truth).item()])
        if nimp < OTLIM:
            bures_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                     ((X[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.)
            logging.info(f"batchsize = {batchsize}, MAE = {bures_scores['MAE'][-1]}, OT = {bures_scores['OT'][-1]}")
        else:
            logging.info(f"batchsize = {batchsize}, MAE = {bures_scores['MAE'][-1]}")
            
        data["imp"]["bures"].append(NAs[mask.bool()].detach().cpu().numpy())
        
        
        logging.info("Linear Round Robin Imputation")
                    
        n, d = X_true.shape

        models = {}
        
        for i in range(d):
            ## predict the ith variable using d-1 others
            models[i] = torch.nn.Linear(d-1, 1).to(device)
        
        linear_rr_imputer = RRimputer(models, max_iter = args.max_iter,
                                            niter = args.rr_niter,
                                            n_pairs = args.n_pairs, 
                                            batchsize = batchsize, 
                                            lr=args.lr, 
                                            weight_decay=args.decay,
                                            order = "random",
                                            eps=gamma, 
                                            opt=torch.optim.Adam, 
                                            scaling=args.scaling)
        
        X = (1 - mask) * X_true.clone() + mask * mean_impute(X_true, mask)
        
        X = linear_rr_imputer.fit_transform(X, mask, 
                                            report_interval=1,
                                            verbose = True,
                                            X_true=X_true).detach()
        
        
        mean_lin, cov_lin = moments(X[M])

        lin_rr_scores['MAE'].append(MAE(X, X_true, mask).item())
        lin_rr_scores['RMSE'].append(RMSE(X, X_true, mask).item())
        lin_rr_scores['bures'].append([((mean_lin - mean_truth)**2).sum().item(), ns_bures(cov_lin, cov_truth).item()])
        if nimp < OTLIM:
            lin_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                     ((X[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.)
            logging.info(f"batchsize = {batchsize}, MAE = {lin_rr_scores['MAE'][-1]}, OT = {lin_rr_scores['OT'][-1]}")
        else:
            logging.info(f"batchsize = {batchsize}, MAE = {lin_rr_scores['MAE'][-1]}")
            
        data["imp"]["lin_rr"].append(NAs[mask.bool()].detach().cpu().numpy())
        
        
        logging.info("MLP Round Robin Imputation")
        
        
        n, d = X_true.shape
        d_ = d-1
        
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
                                   max_iter = args.max_iter, 
                                   niter = args.rr_niter,
                                   n_pairs = args.n_pairs, 
                                   batchsize = batchsize,
                                   lr=args.lr,
                                   weight_decay=args.decay,
                                   order = "random",
                                   eps=gamma, 
                                   opt=torch.optim.Adam, 
                                   scaling=args.scaling)
        
        X = (1 - mask) * X_true.clone() + mask * mean_impute(X_true, mask)
        
        X = mlp_rr_imputer.fit_transform(X.clone(), mask,
                                         report_interval=1,
                                         verbose = True,
                                         X_true=X_true).detach()
        
        mean_mlp, cov_mlp = moments(X[M])

        mlp_rr_scores['MAE'].append(MAE(X, X_true, mask).item())
        mlp_rr_scores['RMSE'].append(RMSE(X, X_true, mask).item())
        mlp_rr_scores['bures'].append([((mean_mlp - mean_truth)**2).sum().item(), ns_bures(cov_mlp, cov_truth).item()])
        if nimp < OTLIM:
            mlp_rr_scores['OT'].append(ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, \
                     ((X[M][:, None] - X_true[M])**2).sum(2).cpu().numpy()) / 2.)
            logging.info(f"batchsize = {batchsize}, MAE = {mlp_rr_scores['MAE'][-1]}, OT = {mlp_rr_scores['OT'][-1]}")
        else:
            logging.info(f"batchsize = {batchsize}, MAE = {mlp_rr_scores['MAE'][-1]}")
            
        data["imp"]["mlp_rr"].append(NAs[mask.bool()].detach().cpu().numpy())
            

    scores = {}
    scores['OT'] = ot_scores
    scores['bures'] = bures_scores
    scores['scikit'] = scikit_scores
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




