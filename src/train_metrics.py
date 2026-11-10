"""Training (in-sample) accuracy of M0-M3 on the full sample, for comparison with the backtest (test) accuracy.
M0/M1: fitted values from the full-sample fits; M2: posterior-median fitted values; M3: one-step-ahead
filtered predictions (each week predicted from the weeks before it, with full-sample parameters), which is
the honest in-sample analogue for a state-space model. Writes reports/tables/train_metrics.json.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, 'src')
import hbnn  # noqa: E402
from models import design, fit_m0  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = hbnn.add_drivers(d)
z = hbnn.attributes(d[d.archived == 0])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
T = int(d.t.max()) + 1
brands = sorted(d.brand.unique())
by_s = {s: g.sort_values('t') for s, g in d.groupby('series_id')}
sids = [s for s in sorted(by_s) if (by_s[s].outlier == 0).sum() >= 8]
G = np.array([brands.index(by_s[s].brand.iloc[0]) for s in sids])
own = np.array([float(by_s[s].own.iloc[0]) for s in sids], np.float32)
arr = hbnn.build(d, sids, lambda s, a: a.outlier == 0, T, z)
res = {m: [] for m in ['M0', 'M1', 'M2', 'M3']}


def wm(y, yh):
    return float(np.abs(y - yh).sum() / y.sum() * 100)


def r2(ly, lyh):
    return float(1 - ((ly - lyh) ** 2).sum() / ((ly - ly.mean()) ** 2).sum())


m1 = pickle.load(open('reports/tables/m1_full_posterior.pkl', 'rb'))
m1idx = {m[0]: j for j, m in enumerate(m1['meta'])}
f2 = hbnn.fit('M2', arr, G, own, len(brands), steps=cfg['svi_steps'])
p2 = hbnn.posterior(f2, n=300)
f3 = hbnn.fit('M3', arr, G, own, len(brands), steps=cfg['svi_steps'])
p3 = hbnn.posterior(f3, n=300)
dr2, _, _ = hbnn.predictive(f2, p2, arr, arr['M_tr'])
dr3, _, _ = hbnn.predictive(f3, p3, arr, arr['M_tr'])     # filter updated on every training week
for j, s in enumerate(sids):
    a = by_s[s][by_s[s].outlier == 0]
    X, _ = design(a)
    y, ly = a.Volume_Sales.values, a.lnV.values
    m0 = fit_m0(X, ly)
    lyh = {'M0': X @ m0['beta']}
    if s in m1idx:
        jj = m1idx[s]
        B = np.median(m1['post']['B'][:, jj], 0)
        lyh['M1'] = X @ B
    tt = a.t.values
    lyh['M2'] = np.median(dr2[:, j, tt], 0) + arr['ybar'][j]
    lyh['M3'] = np.median(dr3[:, j, tt], 0) + arr['ybar'][j]
    for m, v in lyh.items():
        res[m].append({'series_id': s, 'WMAPE': wm(y, np.exp(v)), 'R2_log': r2(ly, v),
                       'resid_sd': float(np.std(ly - v))})
out = {m: {'median_WMAPE': float(np.median([r['WMAPE'] for r in v])),
           'median_R2_log': float(np.median([r['R2_log'] for r in v])),
           'median_resid_sd': float(np.median([r['resid_sd'] for r in v])), 'n_series': len(v)}
       for m, v in res.items()}
json.dump(out, open('reports/tables/train_metrics.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
