"""M2 (HBNN, static elasticities) and M3 (HBNN-KF, Kalman-filtered dynamic base-price elasticity).

Both models use the Equation 1 design of models.py (same 19 columns, same centring), so that
differences in accuracy reflect estimation rather than specification.

Coefficient hierarchy (raw scale):   r_j = mu_brand[g(j)] + pad(NN_phi(z_j)) + tau * e_j
  z_j      PPG attributes: brand, pack-size band, price tier, retailer (one-hot, 17 dims)
  NN_phi   Bayesian MLP 17 -> 8 (tanh) -> 6, giving offsets for base price, promo price ratio
           and the four mechanic-support coefficients
Sign constraints: base-price and promotional-price elasticities = -softplus(raw) (< 0);
                  media coefficient = softplus(raw) (> 0), own brands only.
M3 adds a scalar state per series for the base-price elasticity,
  beta_jt = beta_j + eta_jt,   eta_jt = rho * eta_j,t-1 + w_jt,   w_jt ~ N(0, q_jt),
  q_jt = (1 - rho^2) s_j^2 exp(tanh(MLP_psi(d_jt))),  d_jt = [|d ln TPD|, competitor promo depth, sentiment]
  (s_j = stationary sd of the state, hierarchical HalfNormal(0.3) prior, so the elasticity cannot drift
   without bound for series whose base price barely varies)
and integrates eta out with a Kalman filter written in jax.lax.scan, so the filter's marginal
likelihood is differentiated and trained jointly with the rest of the network (SVI).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal
import optax

from models import design, COLS

jax.config.update('jax_enable_x64', False)
P = len(COLS)
I_BP, I_PR, I_A = COLS.index('lnBP'), COLS.index('lnPR'), COLS.index('lnA')
ENC_IDX = np.array([COLS.index(c) for c in ['lnBP', 'lnPR', 's_tpr', 's_fo', 's_do', 's_fd']])
TAU_SCALE = np.full(P, 0.5); TAU_SCALE[0] = 1.0
H_ENC, H_Q = 8, 4


# ----------------------------------------------------------------------------- data
def attributes(d):
    """Series attribute matrix z (one-hot brand, pack band, tier, retailer)."""
    first = d.groupby('series_id').first()
    med = d.groupby('brand').Price_per_Volume.median()
    tier = med.map(lambda p: 'value' if p < 2.0 else ('mid' if p < 3.0 else 'premium'))
    cols = []
    for name, vals in [('brand', first.brand), ('pack', first.PackSize.astype(str)),
                       ('tier', first.brand.map(tier)), ('market', first.market)]:
        for lev in sorted(vals.unique()):
            cols.append((vals == lev).astype(float).rename(f'{name}={lev}'))
    return pd.concat(cols, axis=1)


def add_drivers(d):
    """Drivers of elasticity change for the M3 process noise (standardised)."""
    d = d.sort_values(['series_id', 't']).copy()
    d['dTPD'] = d.groupby('series_id').lnTPD.diff().abs().fillna(0.0)
    d['depth'] = 1.0 - d.promo_ratio
    mw = d.groupby(['market', 't', 'brand']).apply(
        lambda g: pd_avg(g.depth, g.Volume_Sales), include_groups=False).rename('bd').reset_index()
    tot = mw.groupby(['market', 't']).bd.agg(['sum', 'count']).reset_index()
    d = d.merge(mw, on=['market', 't', 'brand'], how='left').merge(tot, on=['market', 't'], how='left')
    d['comp_depth'] = ((d['sum'] - d['bd']) / (d['count'] - 1).clip(lower=1)).fillna(0.0)
    d = d.drop(columns=['bd', 'sum', 'count'])
    for c in ['dTPD', 'comp_depth', 'csi']:
        d[c + '_z'] = (d[c] - d[c].mean()) / d[c].std()
    return d


def pd_avg(x, w):
    return float(np.average(x, weights=w)) if w.sum() > 0 else float(x.mean())


def build(d, sids, train_mask_fn, T, z):
    """Arrays on a J x T weekly grid.  train_mask_fn(series_id, frame) -> boolean Series of training rows."""
    J = len(sids)
    X = np.zeros((J, T, P), np.float32); Y = np.zeros((J, T), np.float32)
    M_tr = np.zeros((J, T), bool); M_all = np.zeros((J, T), bool)
    D = np.zeros((J, T, 3), np.float32); ybar = np.zeros(J, np.float32)
    for j, s in enumerate(sids):
        a = d[d.series_id == s].sort_values('t')
        tr = train_mask_fn(s, a)
        Xtr, means = design(a[tr])
        Xa, _ = design(a, means)
        tt = a.t.values
        ybar[j] = a.lnV.values[tr.values].mean()
        X[j, tt] = Xa; Y[j, tt] = a.lnV.values - ybar[j]
        M_tr[j, tt] = tr.values; M_all[j, tt] = True
        D[j, tt] = a[['dTPD_z', 'comp_depth_z', 'csi_z']].values
    # A column with no variation in any series' training rows (e.g. post_break before the restatement
    # week) carries no information; as in M1, its effect is treated as absent (zero) at prediction time
    # instead of being drawn from the prior.
    for p in range(1, P):
        sd = [X[j, M_tr[j], p].std() if M_tr[j].any() else 0.0 for j in range(J)]
        if max(sd) < 1e-9:
            X[:, :, p] = 0.0
    return dict(X=X, Y=Y, M_tr=M_tr, M_all=M_all, D=D, ybar=ybar,
                Z=z.loc[sids].values.astype(np.float32),
                G=None, own=None)


# ----------------------------------------------------------------------------- model pieces
def mlp(name, x, n_in, n_hid, n_out, w_scale=1.0, out_scale=0.3):
    W1 = numpyro.sample(f'{name}_W1', dist.Normal(0, w_scale).expand([n_in, n_hid]).to_event(2))
    b1 = numpyro.sample(f'{name}_b1', dist.Normal(0, 1).expand([n_hid]).to_event(1))
    W2 = numpyro.sample(f'{name}_W2', dist.Normal(0, out_scale).expand([n_hid, n_out]).to_event(2))
    return jnp.tanh(x @ W1 + b1) @ W2


# Media options (module-level so that fit() signatures stay simple):
#   MEDIA['mode'] = 'pos'  -> media coefficient softplus(raw) > 0 (default, used for contributions)
#                 = 'free' -> unconstrained media coefficient (used to test H0_4 without the constraint
#                             deciding the answer)
#   MEDIA['loc']  = prior mean of the category-level raw media coefficient (prior sensitivity)
MEDIA = {'mode': 'pos', 'loc': 0.0, 'scale': 3.0}


def coefficients(Z, G, own, n_brand):
    J = Z.shape[0]
    loc0 = jnp.zeros(P).at[I_A].set(MEDIA['loc'])
    sc0 = jnp.full(P, 3.0).at[I_A].set(MEDIA['scale'])
    mu0 = numpyro.sample('mu0', dist.Normal(loc0, sc0).to_event(1))
    om = numpyro.sample('omega', dist.HalfNormal(0.5).expand([P]).to_event(1))
    eg = numpyro.sample('e_brand', dist.Normal(0, 1).expand([n_brand, P]).to_event(2))
    mu_g = mu0 + om * eg
    enc = mlp('enc', Z, Z.shape[1], H_ENC, len(ENC_IDX))
    enc_full = jnp.zeros((J, P)).at[:, ENC_IDX].set(enc)
    tau = numpyro.sample('tau', dist.HalfNormal(TAU_SCALE).to_event(1))
    # Centred parameterisation of the series coefficients: most series have long histories, and with a
    # mean-field guide the non-centred form leaves tau weakly identified (tau * e is identified, the split
    # is not), which produced erratic shrinkage in testing.
    r = numpyro.sample('r_series', dist.Normal(mu_g[G] + enc_full, tau).to_event(2))
    b = r.at[:, I_BP].set(-jax.nn.softplus(r[:, I_BP]))
    b = b.at[:, I_PR].set(-jax.nn.softplus(r[:, I_PR]))
    media = jax.nn.softplus(r[:, I_A]) if MEDIA['mode'] == 'pos' else r[:, I_A]
    b = b.at[:, I_A].set(media * own)
    numpyro.deterministic('b', b)
    ms = numpyro.sample('ms', dist.Normal(jnp.log(0.1), 1.0))
    ss = numpyro.sample('ss', dist.HalfNormal(0.5))
    es = numpyro.sample('e_sigma', dist.Normal(0, 1).expand([J]).to_event(1))
    sigma = jnp.exp(ms + ss * es)
    numpyro.deterministic('sigma', sigma)
    return b, sigma


def m2_model(X, Y, M, Z, G, own, n_brand):
    b, sigma = coefficients(Z, G, own, n_brand)
    mu = jnp.einsum('jtp,jp->jt', X, b)
    ll = dist.Normal(mu, sigma[:, None]).log_prob(Y)
    numpyro.factor('lik', jnp.sum(jnp.where(M, ll, 0.0)))


# Prior on the elasticity state. Price sensitivity in a mature category is expected to move over quarters,
# not weeks: rho ~ Beta(40, 1.5) (mean 0.96) and stationary sd s0 ~ HalfNormal(0.15) give weekly innovations
# of about 0.03 elasticity units. A looser prior (rho ~ Beta(15, 2), s0 ~ HalfNormal(0.3), s_series 0.3)
# produced implausible week-to-week swings of several units and is kept only as a sensitivity check.
STATE_PRIOR = {'rho': (40.0, 1.5), 's0': 0.15, 's_series': 0.2}


def state_noise(D, J, rho, dynamic=True):
    """Process variances of the two states, parameterised through their stationary standard deviations.
      level      : q_l  = (1 - rho_l^2) * s_lj^2,                   s_lj = s_l0 * exp(s_ls * e_lj)
      elasticity : q_jt = (1 - rho^2) * s_j^2 * exp(tanh(MLP(d_jt))), s_j  = s0  * exp(s_s  * e_j)
    s_l0 ~ HalfNormal(0.2) (log-volume units), s0 ~ HalfNormal(0.3) (elasticity units). The bounded network
    term scales the elasticity variance by a factor between e^-1 and e^1 according to the drivers."""
    rho_l = numpyro.sample('rho_l', dist.Beta(30.0, 1.5))
    sl0 = numpyro.sample('s_l0', dist.HalfNormal(0.2))
    sls = numpyro.sample('s_ls', dist.HalfNormal(0.3))
    el = numpyro.sample('e_l', dist.Normal(0, 1).expand([J]).to_event(1))
    slj = sl0 * jnp.exp(sls * el)
    ql = (1 - rho_l ** 2) * slj ** 2
    if not dynamic:
        return rho_l, ql, jnp.zeros((J, D.shape[1])), jnp.zeros(J)
    s0 = numpyro.sample('s0_eta', dist.HalfNormal(STATE_PRIOR['s0']))
    sq = numpyro.sample('s_lq', dist.HalfNormal(STATE_PRIOR['s_series']))
    eq = numpyro.sample('e_lq', dist.Normal(0, 1).expand([J]).to_event(1))
    sj = s0 * jnp.exp(sq * eq)
    g = jnp.tanh(mlp('qnet', D, 3, H_Q, 1, w_scale=0.5, out_scale=0.5)[..., 0])
    return rho_l, ql, (1 - rho ** 2) * (sj ** 2)[:, None] * jnp.exp(g), sj


def kalman(rho, rho_l, ql, q, X, Y, mu, sigma, M_upd, s_eta):
    """Two-state Kalman filter (local level l_t and elasticity deviation eta_t) for all series in parallel:
        y_t = mu_t + l_t + x_t eta_t + e_t,  l_t = rho_l l_t-1 + u_t,  eta_t = rho eta_t-1 + w_t.
    Returns, for every week, the one-step predictive mean and variance of (l_t + x_t eta_t), the predicted
    and filtered state moments (for smoothing), and the log density of y_t."""
    x = X[:, :, I_BP]
    J = X.shape[0]
    init = (jnp.zeros(J), jnp.zeros(J), ql / (1 - rho_l ** 2), jnp.zeros(J), s_eta ** 2)

    def step(c, inp):
        m1, m2, p11, p12, p22 = c
        xt, yt, mut, qt, upd = inp
        a1, a2 = rho_l * m1, rho * m2
        b11, b12, b22 = rho_l ** 2 * p11 + ql, rho_l * rho * p12, rho ** 2 * p22 + qt
        f = a1 + xt * a2
        F = b11 + 2 * xt * b12 + xt ** 2 * b22
        S = F + sigma ** 2
        v = yt - mut - f
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + v ** 2 / S)
        k1, k2 = (b11 + xt * b12) / S, (b12 + xt * b22) / S
        n1 = jnp.where(upd, a1 + k1 * v, a1); n2 = jnp.where(upd, a2 + k2 * v, a2)
        n11 = jnp.where(upd, b11 - k1 * k1 * S, b11); n12 = jnp.where(upd, b12 - k1 * k2 * S, b12)
        n22 = jnp.where(upd, b22 - k2 * k2 * S, b22)
        return (n1, n2, n11, n12, n22), (f, F, a1, a2, b11, b12, b22, n1, n2, n11, n12, n22, ll)

    _, o = jax.lax.scan(step, init, (x.T, Y.T, mu.T, q.T, M_upd.T))
    return tuple(z.T for z in o)


def m3_model(X, Y, M, Z, G, own, n_brand, D, dynamic=True):
    b, sigma = coefficients(Z, G, own, n_brand)
    rho = numpyro.sample('rho', dist.Beta(*STATE_PRIOR['rho'])) if dynamic else jnp.float32(0.9)
    rho_l, ql, q, sj = state_noise(D, X.shape[0], rho, dynamic)
    numpyro.deterministic('q', q); numpyro.deterministic('ql', ql); numpyro.deterministic('rho_l_', rho_l)
    numpyro.deterministic('s_eta', sj)
    mu = jnp.einsum('jtp,jp->jt', X, b)
    ll = kalman(rho, rho_l, ql, q, X, Y, mu, sigma, M, sj)[-1]
    numpyro.factor('kf', jnp.sum(jnp.where(M, ll, 0.0)))


def m3l_model(X, Y, M, Z, G, own, n_brand, D):
    """M3L: local level only, static elasticities (used to isolate elasticity dynamics in RQ2)."""
    m3_model(X, Y, M, Z, G, own, n_brand, D, dynamic=False)


# ----------------------------------------------------------------------------- fit / predict
def fit(kind, arr, G, own, n_brand, steps=4000, lr=0.01, seed=640):
    model = MODELS[kind]
    kw = dict(X=arr['X'], Y=arr['Y'], M=arr['M_tr'], Z=arr['Z'], G=G, own=own, n_brand=n_brand)
    if kind != 'M2':
        kw['D'] = arr['D']
    guide = AutoNormal(model, init_scale=0.05)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(optax.exponential_decay(lr, steps, 0.1)))
    svi = SVI(model, guide, opt, Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, progress_bar=False, **kw)
    return dict(kind=kind, guide=guide, params=res.params, losses=np.asarray(res.losses), kw=kw)


def posterior(f, n=300, seed=1):
    sites = ['b', 'sigma'] + (['q', 'ql', 'rho_l_', 's_eta'] if f['kind'] != 'M2' else []) + \
        (['rho'] if f['kind'] == 'M3' else [])
    pred = Predictive(f['guide'], params=f['params'], num_samples=n, return_sites=None)
    lat = pred(jax.random.PRNGKey(seed), **f['kw'])
    model = MODELS[f['kind']]
    det = Predictive(model, posterior_samples=lat, return_sites=sites)(jax.random.PRNGKey(seed + 1), **f['kw'])
    return {k: np.asarray(v) for k, v in det.items()}


def _rho(post, n):
    return jnp.asarray(post['rho']) if 'rho' in post else jnp.full(n, 0.9, jnp.float32)


def predictive(f, post, arr, M_upd, seed=2):
    """Predictive draws of log volume (centred) on the full grid and log predictive densities.
    For M3 / M3L, M_upd controls which weeks update the filter (training only -> multi-step forecasts;
    training + earlier test weeks -> one-step-ahead leave-future-out densities)."""
    b, sigma = jnp.asarray(post['b']), jnp.asarray(post['sigma'])
    X, Y = jnp.asarray(arr['X']), jnp.asarray(arr['Y'])
    mu = jnp.einsum('jtp,sjp->sjt', X, b)
    if f['kind'] == 'M2':
        mean, var = mu, jnp.broadcast_to(sigma[:, :, None] ** 2, mu.shape)
        eta_m = jnp.zeros_like(mu)
    else:
        n = b.shape[0]
        run = jax.vmap(lambda r, rl, ql, q, m, s, se: kalman(r, rl, ql, q, X, Y, m, s, jnp.asarray(M_upd), se))
        o = run(_rho(post, n), jnp.asarray(post['rho_l_']), jnp.asarray(post['ql']), jnp.asarray(post['q']),
                mu, sigma, jnp.asarray(post['s_eta']))
        mean = mu + o[0]
        var = o[1] + sigma[:, :, None] ** 2
        eta_m = o[3]
    ll = -0.5 * (jnp.log(2 * jnp.pi * var) + (Y[None] - mean) ** 2 / var)
    draws = mean + jnp.sqrt(var) * jax.random.normal(jax.random.PRNGKey(seed), mean.shape)
    return np.asarray(draws), np.asarray(ll), np.asarray(eta_m)


def smoothed_states(f, post, arr):
    """Rauch-Tung-Striebel smoothed states for every posterior draw (M3 / M3L).
    Returns (elasticity path beta_j + eta_jt, level path l_jt), each draws x J x T."""
    X, Y = jnp.asarray(arr['X']), jnp.asarray(arr['Y'])
    M = jnp.asarray(arr['M_tr'])

    def one(b, s, r, rl, ql, q, se):
        mu = jnp.einsum('jtp,jp->jt', X, b)
        (_, _, a1, a2, b11, b12, b22, f1, f2, f11, f12, f22, _) = kalman(r, rl, ql, q, X, Y, mu, s, M, se)

        def back(c, inp):
            s1, s2, s11, s12, s22 = c
            m1, m2, p11, p12, p22, n1, n2, q11, q12, q22 = inp      # filtered t, predicted t+1
            det = q11 * q22 - q12 ** 2
            i11, i12, i22 = q22 / det, -q12 / det, q11 / det
            # C = P_f F' P_p^-1 with F = diag(rl, r)
            c11 = rl * (p11 * i11 + p12 * i12)
            c12 = rl * (p11 * i12 + p12 * i22)
            c21 = r * (p12 * i11 + p22 * i12)
            c22 = r * (p12 * i12 + p22 * i22)
            d1, d2 = s1 - n1, s2 - n2
            o1, o2 = m1 + c11 * d1 + c12 * d2, m2 + c21 * d1 + c22 * d2
            e11, e12, e22 = s11 - q11, s12 - q12, s22 - q22
            o11 = p11 + c11 * (c11 * e11 + c12 * e12) + c12 * (c11 * e12 + c12 * e22)
            o12 = p12 + c11 * (c21 * e11 + c22 * e12) + c12 * (c21 * e12 + c22 * e22)
            o22 = p22 + c21 * (c21 * e11 + c22 * e12) + c22 * (c21 * e12 + c22 * e22)
            return (o1, o2, o11, o12, o22), (o1, o2)
        init = (f1[:, -1], f2[:, -1], f11[:, -1], f12[:, -1], f22[:, -1])
        rev = lambda z: z.T[::-1]  # noqa: E731
        xs = (rev(f1[:, :-1]), rev(f2[:, :-1]), rev(f11[:, :-1]), rev(f12[:, :-1]), rev(f22[:, :-1]),
              rev(a1[:, 1:]), rev(a2[:, 1:]), rev(b11[:, 1:]), rev(b12[:, 1:]), rev(b22[:, 1:]))
        _, (l_rev, e_rev) = jax.lax.scan(back, init, xs)
        lev = jnp.concatenate([l_rev[::-1], f1[:, -1:].T], 0).T
        eta = jnp.concatenate([e_rev[::-1], f2[:, -1:].T], 0).T
        return b[:, I_BP][:, None] + eta, lev
    n = post['b'].shape[0]
    el, lev = jax.vmap(one)(jnp.asarray(post['b']), jnp.asarray(post['sigma']), _rho(post, n),
                            jnp.asarray(post['rho_l_']), jnp.asarray(post['ql']), jnp.asarray(post['q']),
                            jnp.asarray(post['s_eta']))
    return np.asarray(el), np.asarray(lev)


def smoothed_elasticity(f, post, arr):
    return smoothed_states(f, post, arr)[0]


MODELS = {'M2': m2_model, 'M3': m3_model, 'M3L': m3l_model}
