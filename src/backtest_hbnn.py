"""Stage 5/6: rolling-origin backtest of M2 (HBNN) and M3 (HBNN-KF).
Uses exactly the origins, horizon, truncation folds and training-row rules of backtest.py, so the
results can be paired series by series with M0 and M1.
Writes reports/tables/backtest_forecasts_hbnn.csv (same columns as backtest_forecasts.csv) and
reports/tables/lfo_density.csv (log predictive densities for RQ2: multi-step and one-step-ahead).
Usage: python src/backtest_hbnn.py [quick]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, 'src')
import hbnn  # noqa: E402
from models import crps_samples  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
STEPS = cfg.get('svi_steps', 4000)
NDRAW = cfg.get('svi_draws', 300)
QUICK = 'quick' in sys.argv[1:]
ONLY = [a for a in sys.argv[1:] if a in ('M2', 'M3', 'M3L')]   # e.g. `python src/backtest_hbnn.py M3 M3L`
KINDS = tuple(ONLY) or ('M2', 'M3', 'M3L')        # M3L (level-only) is fitted in the full scenario only (RQ2)
if QUICK:
    STEPS = 1500

d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = hbnn.add_drivers(d)
z_all = hbnn.attributes(d[d.archived == 0])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
T = int(d.t.max()) + 1
sids_all = sorted(d.series_id.unique())
brands = sorted(d.brand.unique())
natural_sparse = set(d[d.sparse_flag == 1].series_id.unique())
H, TR, MIN_TRAIN = cfg['horizon'], cfg['truncate_weeks'], 8
rich = [s for s in sids_all if s not in natural_sparse]
folds = np.array_split(np.random.default_rng(1).permutation(rich), 5)   # identical to backtest.py
by_s = {s: g.sort_values('t') for s, g in d.groupby('series_id')}

rows, dens, diag = [], [], []


def train_rows(a, origin, trunc):
    tr = a.t < origin
    if trunc:
        tmax = a.t[tr].max()
        tr &= a.t >= tmax - TR + 1
    return tr & (a.outlier == 0)


def run(origin, truncate=frozenset(), tag='full', kinds=None):
    kinds = [k for k in (kinds or KINDS) if not (tag == 'trunc' and k == 'M3L')]
    sids = [s for s in sids_all if train_rows(by_s[s], origin, s in truncate).sum() >= MIN_TRAIN]
    arr = hbnn.build(d, sids, lambda s, a: train_rows(a, origin, s in truncate), T, z_all)
    G = np.array([brands.index(by_s[s].brand.iloc[0]) for s in sids])
    own = np.array([float(by_s[s].own.iloc[0]) for s in sids], np.float32)
    test = np.zeros_like(arr['M_all'])
    test[:, origin:origin + H] = arr['M_all'][:, origin:origin + H]
    keep = []
    for j, s in enumerate(sids):
        if not test[j].any():
            continue
        if tag != 'full' and (s in natural_sparse or s not in truncate):
            continue
        keep.append(j)
    for kind in kinds:
        t0 = time.time()
        f = hbnn.fit(kind, arr, G, own, len(brands), steps=STEPS, seed=cfg['seed'] + origin)
        post = hbnn.posterior(f, n=NDRAW, seed=origin)
        draws, ll_multi, _ = hbnn.predictive(f, post, arr, arr['M_tr'], seed=origin)
        if kind != 'M2':
            _, ll_one, _ = hbnn.predictive(f, post, arr, arr['M_tr'] | test, seed=origin)
        else:
            ll_one = ll_multi
        lpd_m = np.log(np.mean(np.exp(ll_multi - ll_multi.max(0)), 0)) + ll_multi.max(0)
        lpd_1 = np.log(np.mean(np.exp(ll_one - ll_one.max(0)), 0)) + ll_one.max(0)
        L = f['losses']
        diag.append({'origin': origin, 'scenario': tag, 'model': kind, 'secs': round(time.time() - t0, 1),
                     'elbo_first': float(L[0]), 'elbo_last': float(L[-1]),
                     'elbo_rel_change_last10pct': float(abs(L[-len(L) // 10:].mean() - L[-2 * len(L) // 10:-len(L) // 10].mean())
                                                        / abs(L[-1])),
                     'rho_median': float(np.median(post['rho'])) if kind == 'M3' else None,
                     's_eta_median': float(np.median(post['s_eta'])) if kind == 'M3' else None,
                     's_eta_max_series': float(np.median(post['s_eta'], 0).max()) if kind == 'M3' else None})
        for j in keep:
            s = sids[j]
            tt = np.where(test[j])[0]
            a = by_s[s].set_index('t')
            yv = a.loc[tt, 'Volume_Sales'].values
            vol_tr = np.exp(a.loc[np.where(arr['M_tr'][j])[0], 'lnV'].values)
            scale = np.abs(np.diff(vol_tr)).mean() if len(vol_tr) > 1 else vol_tr.mean()
            D = np.exp(draws[:, j, tt] + arr['ybar'][j])
            med = np.median(D, 0); lo, hi = np.quantile(D, .05, 0), np.quantile(D, .95, 0)
            cr = crps_samples(D, yv)
            for k, t in enumerate(tt):
                rows.append({'origin': origin, 'scenario': tag, 'series_id': s, 't': int(t), 'model': kind,
                             'y': yv[k], 'yhat': med[k], 'lo': lo[k], 'hi': hi[k], 'crps': cr[k],
                             'scale': scale, 'n_train': int(arr['M_tr'][j].sum())})
                dens.append({'origin': origin, 'scenario': tag, 'series_id': s, 't': int(t), 'model': kind,
                             'lpd_multi': float(lpd_m[j, t]), 'lpd_1step': float(lpd_1[j, t])})
        print(f'  {tag} origin {origin} {kind}: {time.time() - t0:.0f}s, ELBO {L[0]:.0f} -> {L[-1]:.0f}', flush=True)


if __name__ == '__main__':
    t0 = time.time()
    origins = cfg['backtest_origins'][-1:] if QUICK else cfg['backtest_origins']
    for o in origins:
        run(o)
        for f in (folds[:1] if QUICK else folds):
            run(o, frozenset(f), 'trunc')
        print(f'origin {o} done, {time.time() - t0:.0f}s', flush=True)
    sfx = '_quick' if QUICK else ''
    for name, new in [('backtest_forecasts_hbnn', rows), ('lfo_density', dens), ('svi_diagnostics', diag)]:
        path = f'reports/tables/{name}{sfx}.csv'
        new = pd.DataFrame(new)
        if ONLY:   # replace only the refitted model's rows
            try:
                old = pd.read_csv(path)
                new = pd.concat([old[~old.model.isin(ONLY)], new], ignore_index=True)
            except FileNotFoundError:
                pass
        new.to_csv(path, index=False)
    print('total', round(time.time() - t0), 's')
