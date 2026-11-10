"""Recovery study on the synthetic panel: can M2 and M3 recover the known elasticities?

Reports bias, RMSE and 90% interval coverage of
  - the static base-price elasticity (M2) on series whose true elasticity is static, and
  - the weekly elasticity path (M3, RTS-smoothed) on series whose true elasticity moves,
and compares M3's path error with M2's constant estimate on the same series.
Usage: python src/recovery.py      (run src/synthetic.py first)
"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
sys.path.insert(0, 'src')
import hbnn  # noqa: E402
from models import COLS  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
d = pd.read_csv('data/synthetic/panel_synthetic.csv', parse_dates=['week'])
truth = pd.read_csv('data/synthetic/truth.csv')
d = hbnn.add_drivers(d)
z = hbnn.attributes(d)
T = int(d.t.max()) + 1
brands = sorted(d.brand.unique())
sids = sorted(d.series_id.unique())
by_s = {s: g for s, g in d.groupby('series_id')}
G = np.array([brands.index(by_s[s].brand.iloc[0]) for s in sids])
own = np.array([float(by_s[s].own.iloc[0]) for s in sids], np.float32)
arr = hbnn.build(d, sids, lambda s, a: a.outlier == 0, T, z)
iBP = COLS.index('lnBP')
res = {}
f2 = hbnn.fit('M2', arr, G, own, len(brands), steps=cfg.get('svi_steps', 8000))
p2 = hbnn.posterior(f2, n=400)
f3 = hbnn.fit('M3', arr, G, own, len(brands), steps=cfg.get('svi_steps', 8000))
p3 = hbnn.posterior(f3, n=400)
paths = hbnn.smoothed_elasticity(f3, p3, arr)
rows = []
for j, s in enumerate(sids):
    tr = truth[truth.series_id == s].set_index('t')
    tt = tr.index.values
    b2 = p2['b'][:, j, iBP]
    pm = np.median(paths[:, j, tt], 0)
    lo, hi = np.quantile(paths[:, j, tt], [.05, .95], axis=0)
    rows.append({'series_id': s, 'kind': tr.kind.iloc[0], 'n': len(tt),
                 'true_mean': tr.beta_t.mean(),
                 'M2_err': np.median(b2) - tr.beta_t.mean(),
                 'M2_cover': float(np.quantile(b2, .05) <= tr.beta_t.mean() <= np.quantile(b2, .95)),
                 'M2_path_rmse': float(np.sqrt(np.mean((np.median(b2) - tr.beta_t.values) ** 2))),
                 'M3_path_rmse': float(np.sqrt(np.mean((pm - tr.beta_t.values) ** 2))),
                 'M3_path_bias': float(np.mean(pm - tr.beta_t.values)),
                 'M3_path_cover': float(np.mean((tr.beta_t.values >= lo) & (tr.beta_t.values <= hi)))})
r = pd.DataFrame(rows)
r.to_csv('reports/tables/recovery_by_series.csv', index=False)
for k, g in r.groupby(r.kind.where(r.kind == 'static', 'dynamic')):
    res[k] = {'n_series': int(len(g)),
              'M2_static_bias': float(g.M2_err.mean()), 'M2_static_rmse': float(np.sqrt((g.M2_err ** 2).mean())),
              'M2_static_cover90': float(g.M2_cover.mean()),
              'M2_path_rmse_median': float(g.M2_path_rmse.median()),
              'M3_path_rmse_median': float(g.M3_path_rmse.median()),
              'M3_path_bias_mean': float(g.M3_path_bias.mean()),
              'M3_path_cover90_mean': float(g.M3_path_cover.mean()),
              'share_M3_lower_path_rmse': float((g.M3_path_rmse < g.M2_path_rmse).mean())}
mech_true = truth.groupby('series_id')[['theta_TPR', 'theta_FO', 'theta_DO', 'theta_F&D']].first().loc[sids]
for m, c in zip(['TPR', 'FO', 'DO', 'F&D'], ['s_tpr', 's_fo', 's_do', 's_fd']):
    est = np.median(p3['b'][:, :, COLS.index(c)], 0)
    res[f'theta_{m}'] = {'true_mean': float(mech_true[f'theta_{m}'].mean()), 'est_mean': float(est.mean()),
                         'rmse': float(np.sqrt(np.mean((est - mech_true[f'theta_{m}'].values) ** 2)))}
res['rho_M3'] = [float(x) for x in np.quantile(p3['rho'], [.05, .5, .95])]
json.dump(res, open('reports/tables/recovery_results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

ex = r[r.kind != 'static'].sort_values('n', ascending=False).head(2).series_id.tolist() + \
    r[r.kind == 'static'].head(1).series_id.tolist()
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=False)
for ax, s in zip(axes, ex):
    j = sids.index(s)
    tr = truth[truth.series_id == s].set_index('t')
    tt = tr.index.values
    lo, md, hi = np.quantile(paths[:, j, tt], [.05, .5, .95], axis=0)
    ax.fill_between(tt, lo, hi, color='#1f4e79', alpha=.2, label='M3 90% band')
    ax.plot(tt, md, color='#1f4e79', lw=1.2, label='M3 smoothed')
    ax.plot(tt, tr.beta_t, color='k', ls='--', lw=1, label='True')
    ax.axhline(np.median(p2['b'][:, j, iBP]), color='#c0504d', lw=1, label='M2 static')
    ax.set_title(f'{s} ({tr.kind.iloc[0]})', fontsize=8.5, loc='left')
    ax.set_xlabel('Week')
axes[0].set_ylabel('Base-price elasticity')
axes[0].legend(frameon=False, fontsize=6.5, loc='lower left')
fig.tight_layout()
fig.savefig('reports/figures/fig19_recovery.png', dpi=200)
plt.close()
