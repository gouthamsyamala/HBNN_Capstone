"""Baseline models.
M0: independent series-level OLS log-log regressions.
M1: hierarchical Bayesian log-log regression (series -> brand -> category), static coefficients,
    estimated with a Gibbs sampler (conjugate normal / inverse-gamma).
"""
import numpy as np
import pandas as pd

FEATS = ['lnBP', 'lnPR', 's_tpr', 's_fo', 's_do', 's_fd', 'lnTPD', 'lnCP', 'lnA',
         'sin1', 'cos1', 'sin2', 'cos2', 'hol_thanks', 'hol_xmas', 'hol_easter', 'trend', 'post_break']
CENTER = ['lnBP', 'lnTPD', 'lnCP', 'lnA', 'trend']   # centred on the series' training mean
REDUCED = ['lnBP', 'lnPR', 'lnTPD', 'trend']          # M0 specification when history is short
COLS = ['const'] + FEATS


def design(df, means=None):
    x = df.copy()
    x['trend'] = x['t'] / 52.0
    if means is None:
        means = x[CENTER].mean()
    for c in CENTER:
        x[c] = x[c] - means[c]
    X = np.column_stack([np.ones(len(x))] + [x[f].values for f in FEATS])
    return X, means


# --------------------------------------------------------------------------- M0
def fit_m0(X, y):
    n = len(y)
    keep = np.arange(X.shape[1])
    if n < 3 * X.shape[1]:
        keep = np.array([0] + [1 + FEATS.index(f) for f in REDUCED])
    Xk = X[:, keep]
    XtX_inv = np.linalg.pinv(Xk.T @ Xk)
    b = XtX_inv @ Xk.T @ y
    rank = np.linalg.matrix_rank(Xk)
    dof = max(n - rank, 1)
    s2 = float(((y - Xk @ b) ** 2).sum() / dof)
    beta = np.zeros(X.shape[1])
    beta[keep] = b
    return {'beta': beta, 'keep': keep, 'XtX_inv': XtX_inv, 's2': s2, 'dof': dof}


def predict_m0(m, X, ndraw=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    Xk = X[:, m['keep']]
    mu = Xk @ m['beta'][m['keep']]
    var = m['s2'] * (1 + np.einsum('ij,jk,ik->i', Xk, m['XtX_inv'], Xk))
    draws = mu + np.sqrt(var) * rng.standard_t(m['dof'], size=(ndraw, len(mu)))
    return np.exp(draws)


# --------------------------------------------------------------------------- M1
def fit_m1(series, groups, n_iter=2000, burn=1000, thin=2, seed=640):
    """series: list of (X, y); groups: list of brand index per series.
    Coefficients whose column has no variation in a series are treated as absent for that series
    (they do not inform the brand / category hyper-parameters)."""
    rng = np.random.default_rng(seed)
    J, p = len(series), series[0][0].shape[1]
    G = int(max(groups)) + 1
    groups = np.asarray(groups)
    XtX = [X.T @ X for X, _ in series]
    Xty = [X.T @ y for X, y in series]
    n = np.array([len(y) for _, y in series])
    mask = np.array([[True] + list(X[:, 1:].std(0) > 1e-9) for X, _ in series])  # J x p
    # priors
    a_t, b_t = np.full(p, 1.0), np.full(p, 0.05)
    b_t[0] = 1.0                       # intercept varies widely between series
    a_s, b_s = 2.0, 0.05
    prior_var0 = 100.0
    # init with ridge per series
    B = np.array([np.linalg.solve(XtX[j] + 0.1 * np.eye(p), Xty[j]) for j in range(J)])
    s2 = np.full(J, 0.05)
    mu_g = np.array([B[groups == g].mean(0) if (groups == g).any() else B.mean(0) for g in range(G)])
    mu0 = B.mean(0)
    tau2 = np.full(p, 0.1); tau2[0] = 2.0
    om2 = np.full(p, 0.1); om2[0] = 2.0
    keep = {'B': [], 's2': [], 'mu_g': [], 'mu0': []}
    for it in range(n_iter):
        Tinv = 1.0 / tau2
        for j in range(J):
            prec = XtX[j] / s2[j] + np.diag(Tinv)
            L = np.linalg.cholesky(prec)
            rhs = Xty[j] / s2[j] + Tinv * mu_g[groups[j]]
            m = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            B[j] = m + np.linalg.solve(L.T, rng.standard_normal(p))
            X, y = series[j]
            rss = float(((y - X @ B[j]) ** 2).sum())
            s2[j] = 1.0 / rng.gamma(a_s + n[j] / 2, 1.0 / (b_s + rss / 2))
        for g in range(G):
            idx = groups == g
            cnt = mask[idx].sum(0)
            sm = (B[idx] * mask[idx]).sum(0)
            prec = cnt / tau2 + 1.0 / om2
            mu_g[g] = (sm / tau2 + mu0 / om2) / prec + rng.standard_normal(p) / np.sqrt(prec)
        dead = ~mask.any(0)              # coefficient not identified by any series in this sample
        mu_g[:, dead] = 0.0
        prec0 = G / om2 + 1.0 / prior_var0
        mu0 = (mu_g.sum(0) / om2) / prec0 + rng.standard_normal(p) / np.sqrt(prec0)
        dev = ((B - mu_g[groups]) ** 2) * mask
        mu0[dead] = 0.0
        tau2 = 1.0 / rng.gamma(a_t + mask.sum(0) / 2, 1.0 / (b_t + dev.sum(0) / 2))
        tau2[dead] = 1e-6
        om2 = 1.0 / rng.gamma(a_t + G / 2, 1.0 / (b_t + ((mu_g - mu0) ** 2).sum(0) / 2))
        if it >= burn and (it - burn) % thin == 0:
            keep['B'].append(B.copy()); keep['s2'].append(s2.copy())
            keep['mu_g'].append(mu_g.copy()); keep['mu0'].append(mu0.copy())
    return {k: np.array(v) for k, v in keep.items()} | {'mask': mask, 'groups': groups}


def predict_m1(post, j, X, rng=None):
    rng = rng or np.random.default_rng(1)
    B = post['B'][:, j, :].copy()
    # a coefficient the series' own history cannot identify is predicted from its brand-level mean
    absent = ~post['mask'][j]
    B[:, absent] = post['mu_g'][:, post['groups'][j], :][:, absent]
    s = np.sqrt(post['s2'][:, j])
    mu = B @ X.T
    return np.exp(mu + s[:, None] * rng.standard_normal(mu.shape))


# --------------------------------------------------------------------------- metrics
def crps_samples(draws, y):
    """Sample-based CRPS for each column of draws (ndraw x T) against y (T)."""
    d = np.sort(draws, axis=0)
    m = d.shape[0]
    t1 = np.abs(d - y).mean(0)
    w = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    t2 = (w * d).sum(0) / (m * m)
    return t1 - t2
