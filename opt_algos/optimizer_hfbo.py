#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate HMF-Opt (Hierarchical Multi-Fidelity Optimization) on the Data-Model benchmark using
an explicit hierarchical GP with logistic scaling and residual models.
"""
import argparse
import os
import sys
import warnings

# ensure project root on PYTHONPATH
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

import numpy as np
import torch
import gpytorch
from scipy.optimize import curve_fit
import benchmarks
import misc_utils as mscu

# FLOPS per model scale from eval_ei_optimizer
FLOPS = {
    100: 1,
    2: 2090524455 / 161264981936,
    6: 5211827866 / 161264981936,
    15: 12069704997 / 161264981936,
    30: 23823782173 / 161264981936,
    50: 34933622501 / 161264981936,
    70: 48105020743 / 161264981936,
}

# logistic function for scaling
def logistic(z, a, b, z0, s):
    return a + b / (1.0 + np.exp(-(z - z0) / s))

class LogisticScaler:
    """Fit a logistic scaling curve rho(z) = a + b/(1+exp(-(z-z0)/s))."""
    def __init__(self):
        self.params = None  # (a, b, z0, s)
    def fit(self, zs, ratios):
        if len(zs) < 2:
            # default to identity
            self.params = np.array([1.0, 0.0, 0.0, 1.0])
        else:
            p0 = [np.min(ratios), np.max(ratios)-np.min(ratios), np.median(zs), np.std(zs)]
            self.params, _ = curve_fit(logistic, zs, ratios, p0=p0, maxfev=5000)
    def __call__(self, z):
        if self.params is None:
            return 1.0
        a, b, z0, s = self.params
        return logistic(z, a, b, z0, s)

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

class HierarchicalMultiFidelityGP:
    """
    Implements a hierarchical GP: for each adjacent fidelity pair f_i->f_{i+1},
    fits rho_i(z) via logistic scaling and a residual GP delta_i(w).
    Prediction at any fidelity f_j composes through the chain.
    """
    def __init__(self, fidelities, scale_scale, step_scale):
        self.fidelities = sorted(fidelities, key=lambda f: FLOPS[int(f[0]*scale_scale)]*f[1]*step_scale)
        self.n_levels = len(self.fidelities)
        self.scale_scale = scale_scale
        self.step_scale = step_scale
        self.scalers = []       # list of LogisticScaler for each transition
        self.residual_gps = []  # list of ExactGP or None
        self.likelihoods = []
        self.train_data = []    # list of (w, f, y)

    def fit_initial(self, data):
        """Fit scalers and residual GPs from warm-start data."""
        self.train_data = data.copy()
        # group by fidelity
        grouped = {f: [] for f in self.fidelities}
        for w, f, y in self.train_data:
            grouped[f].append((w, y))
        self.scalers.clear(); self.residual_gps.clear(); self.likelihoods.clear()
        # for each transition i->i+1
        for i in range(self.n_levels - 1):
            f_lo, f_hi = self.fidelities[i], self.fidelities[i+1]
            dict_lo = {tuple(w): y for w, y in grouped.get(f_lo, [])}
            dict_hi = {tuple(w): y for w, y in grouped.get(f_hi, [])}
            common_ws = [np.array(w) for w in dict_lo if w in dict_hi]
            zs, ratios, resid_x, resid_y = [], [], [], []
            for w in common_ws:
                y_lo = dict_lo[tuple(w)]; y_hi = dict_hi[tuple(w)]
                z_lo = f_lo[1] * self.step_scale
                ratio = y_hi / (y_lo + 1e-12)
                zs.append(z_lo); ratios.append(ratio)
                resid_x.append(w); resid_y.append(y_hi - ratio * y_lo)
            # fit logistic scaler
            scaler = LogisticScaler()
            scaler.fit(np.array(zs), np.array(ratios))
            self.scalers.append(scaler)
            # fit residual GP if data available
            if len(resid_x) >= 2:
                X = torch.tensor(np.vstack(resid_x), dtype=torch.double)
                y = torch.tensor(resid_y, dtype=torch.double)
                lik = gpytorch.likelihoods.GaussianLikelihood()
                gp = ExactGP(X, y, lik).double()
                gp.train(); lik.train()
                opt = torch.optim.Adam(gp.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
                for _ in range(50): opt.zero_grad(); loss = -mll(gp(X), y); loss.backward(); opt.step()
                gp.eval(); lik.eval()
                self.residual_gps.append(gp); self.likelihoods.append(lik)
            else:
                self.residual_gps.append(None); self.likelihoods.append(None)

    def predict(self, w, f):
        """Return (mean, var) at fidelity f."""
        mu_prev, var_prev = 0.0, 0.0
        for i, f_lo in enumerate(self.fidelities[:-1]):
            z_lo = f_lo[1] * self.step_scale
            rho = self.scalers[i](z_lo)
            # residual
            gp = self.residual_gps[i]
            if gp is not None:
                Xw = torch.tensor(w, dtype=torch.double).unsqueeze(0)
                with torch.no_grad(): post = gp(Xw)
                mu_res, var_res = post.mean.item(), post.variance.item()
            else:
                mu_res, var_res = 0.0, 1e-6
            mu_prev = rho * mu_prev + mu_res
            var_prev = (rho**2) * var_prev + var_res
            if f_lo == f:
                return mu_prev, var_prev
        # if highest fidelity desired
        return mu_prev, var_prev

    def update(self, w, f, y):
        """Add new point and refit only affected transitions."""
        self.train_data.append((w, f, y))
        self.fit_initial(self.train_data)

# ---- Script ----
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate HMF-Opt on the Data-Model benchmark")
    p.add_argument("--metric_index", type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--ini_rand",     type=int,   default=10)
    p.add_argument("--iters",        type=int,   default=15)
    p.add_argument("--num_mc",       type=int,   default=1000)
    p.add_argument("--scale_scale",  type=float, default=100.0)
    p.add_argument("--step_scale",   type=float, default=197.0)
    p.add_argument("--log_file",     type=str,   default="hmf.log")
    return p.parse_args()


def calculate_cost(scale_frac, step_frac, scale_scale, step_scale):
    scale = int(scale_frac * scale_scale)
    steps = int(step_frac * step_scale)
    return FLOPS[scale] * steps


def main():
    args = parse_args()
    logger = mscu.get_logger(filename=args.log_file)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    bm   = benchmarks.DataModelBenchmark(metric_index=args.metric_index)
    func = bm._raw_func_with_model_scale
    raw_func = bm._raw_func

    SCALES    = np.array([2,6,15,30,50,70,100]) / args.scale_scale
    TIMESTEPS = np.array([60,120,197])      / args.step_scale
    fidelities = [(s, t) for s in SCALES for t in TIMESTEPS]

    # warm-start
    data = []
    dim = len(bm.search_space)
    for _ in range(args.ini_rand):
        w = np.random.dirichlet(np.ones(dim))
        s = np.random.choice(SCALES)
        t = np.random.choice(TIMESTEPS)
        z, m = int(t*args.step_scale), int(s*args.scale_scale)
        y_vals = func(z, m, w)
        y = float(y_vals[-1][1]) if isinstance(y_vals, list) else float(y_vals)
        data.append((w, (s,t), y))

    hmf = HierarchicalMultiFidelityGP(fidelities, args.scale_scale, args.step_scale)
    hmf.fit_initial(data)

    history = []
    cum_cost = sum(calculate_cost(f[0],f[1],args.scale_scale,args.step_scale) for _,f,_ in data)
    candidate_ws = [w for w,_,_ in data]
    f_star = fidelities[-1]

    for itr in range(1, args.iters+1):
        mu_vals = [hmf.predict(w,f_star)[0] for w in candidate_ws]
        mu_star = max(mu_vals)
        best_score, best_choice = -np.inf, None
        for f in fidelities:
            cost_f = calculate_cost(f[0],f[1],args.scale_scale,args.step_scale)
            for w in candidate_ws:
                mu, var = hmf.predict(w, f)
                sigma = np.sqrt(var)
                samples = np.random.normal(mu, sigma, size=args.num_mc)
                hyps = []
                for y_samp in samples:
                    vals = [y_samp if np.allclose(w2,w) else hmf.predict(w2,f_star)[0] for w2 in candidate_ws]
                    hyps.append(max(vals))
                kg = float(np.mean(hyps) - mu_star)
                score = kg / (cost_f + 1e-12)
                if score > best_score:
                    best_score, best_choice = score, (w, f)
        w_next, f_next = best_choice
        z_next, m_next = int(f_next[1]*args.step_scale), int(f_next[0]*args.scale_scale)
        y_vals = func(z_next, m_next, w_next)
        y_next = float(y_vals[-1][1]) if isinstance(y_vals, list) else float(y_vals)
        cum_cost += calculate_cost(f_next[0],f_next[1],args.scale_scale,args.step_scale)
        # history.append((w_next, f_next, y_next, best_score, cum_cost))

        y_final = raw_func(196, w_next)
        history.append((w_next, f_next, y_final, best_score, cum_cost))

        logger.info(f"[HMF] Iter {itr} w={w_next} f={f_next} y={y_next:.4f} KG={best_score:.3e} cost={cum_cost:.2f}")
        hmf.update(w_next, f_next, y_next)
        if not any(np.allclose(w_next, w) for w in candidate_ws): candidate_ws.append(w_next)

    best_idx = np.argmax([hmf.predict(w,f_star)[0] for w in candidate_ws])
    best_w = candidate_ws[best_idx]
    best_val, _ = hmf.predict(best_w, f_star)
    print("Final best mixture:", best_w)
    print("Estimated value at full fidelity:", best_val)
    print("Total cost consumed:", cum_cost)
    np.save("hmf_history.npy", np.array(history, dtype=object))

if __name__ == "__main__":
    main()