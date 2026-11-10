"""Stage 4/6: rolling-origin backtest (4 origins x 13-week horizon covering the last 52 weeks).
Scenarios: 'full'  = every series uses all history before the origin;
           'trunc' = rich series in one fold are cut to their last 26 training weeks (artificially sparse)
                     while all other series keep full history (5 folds per origin).
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, 'src')
from models import design, fit_m0, predict_m0, fit_m1, predict_m1, crps_samples  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
rng = np.random.default_rng(cfg['seed'])
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
sids = sorted(d.series_id.unique())
brands = sorted(d.brand.unique())
gidx = {s: brands.index(d[d.series_id == s].brand.iloc[0]) for s in sids}
natural_sparse = set(d[d.sparse_flag == 1].series_id.unique())
H, TR, MIN_TRAIN = cfg['horizon'], cfg['truncate_weeks'], 8
N_ITER, BURN = cfg['gibbs_iter'], cfg['gibbs_burn']
if len(sys.argv) > 1 and sys.argv[1] == 'quick':
    N_ITER, BURN = 600, 300

rows = []


def run(origin, truncate=frozenset(), tag='full'):
    tr = d[(d.t < origin)]
    te = d[(d.t >= origin) & (d.t < origin + H)]
    series, meta = [], []
    for s in sids:
        a = tr[tr.series_id == s]
        if s in truncate:
            a = a[a.t >= a.t.max() - TR + 1]
        a = a[a.outlier == 0]
        b = te[te.series_id == s]
        if len(a) < MIN_TRAIN:
            continue
        X, means = design(a)
        y = a.lnV.values
        vol = np.exp(a.sort_values('t').lnV.values)
        scale = np.abs(np.diff(vol)).mean() if len(vol) > 1 else vol.mean()
        Xt, _ = design(b, means) if len(b) else (None, None)
        series.append((X, y))
        meta.append((s, b, Xt, scale, len(a)))
    post = fit_m1(series, [gidx[m[0]] for m in meta], n_iter=N_ITER, burn=BURN, seed=cfg['seed'] + origin)
    for j, (s, b, Xt, scale, ntr) in enumerate(meta):
        if s in natural_sparse and tag != 'full':
            continue
        if tag == 'trunc' and s not in truncate:
            continue
        if Xt is None:
            continue
        m0 = fit_m0(*series[j])
        dr = {'M0': predict_m0(m0, Xt, rng=rng), 'M1': predict_m1(post, j, Xt, rng=rng)}
        yv = b.Volume_Sales.values
        for mod, D in dr.items():
            med = np.median(D, 0)
            lo, hi = np.quantile(D, .05, 0), np.quantile(D, .95, 0)
            cr = crps_samples(D, yv)
            for k in range(len(yv)):
                rows.append({'origin': origin, 'scenario': tag, 'series_id': s, 't': int(b.t.values[k]),
                             'model': mod, 'y': yv[k], 'yhat': med[k], 'lo': lo[k], 'hi': hi[k],
                             'crps': cr[k], 'scale': scale, 'n_train': ntr})
    return post, meta, series


t0 = time.time()
rich = [s for s in sids if s not in natural_sparse]
folds = np.array_split(np.random.default_rng(1).permutation(rich), 5)
coef_trunc = []
for o in cfg['backtest_origins']:
    run(o)
    for f in folds:
        post, meta, series = run(o, frozenset(f), 'trunc')
        if o == cfg['backtest_origins'][-1]:
            for j, (s, *_rest) in enumerate(meta):
                if s in f:
                    m0 = fit_m0(*series[j])
                    coef_trunc.append({'series_id': s, 'M0_lnBP': float(m0['beta'][1]),
                                       'M0_lnPR': float(m0['beta'][2]), 'M1_lnBP': float(np.median(post['B'][:, j, 1])),
                                       'M1_lnPR': float(np.median(post['B'][:, j, 2])),
                                       'M1_lnBP_lo': float(np.quantile(post['B'][:, j, 1], .05)),
                                       'M1_lnBP_hi': float(np.quantile(post['B'][:, j, 1], .95))})
    print(f'origin {o} done, {time.time() - t0:.0f}s', flush=True)

fc = pd.DataFrame(rows)
fc.to_csv('reports/tables/backtest_forecasts.csv', index=False)
pd.DataFrame(coef_trunc).to_csv('reports/tables/coef_truncated_M1.csv', index=False)
print('rows', len(fc), 'time', time.time() - t0)
