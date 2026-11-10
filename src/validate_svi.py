"""Check the variational (SVI) posterior against the No-U-Turn sampler on a 10% subsample of series.
For M2 and M3, compares series-level base-price elasticity medians and 90% interval widths, and the
residual scale. Writes reports/tables/svi_vs_nuts.json.
"""
import json
import sys

import jax
import numpy as np
import pandas as pd
import yaml
from numpyro.infer import MCMC, NUTS, Predictive

sys.path.insert(0, 'src')
import hbnn  # noqa: E402
from models import COLS  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = hbnn.add_drivers(d)
z = hbnn.attributes(d[d.archived == 0])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
T = int(d.t.max()) + 1
brands = sorted(d.brand.unique())
by_s = {s: g for s, g in d.groupby('series_id')}
alls = [s for s in sorted(by_s) if (by_s[s].outlier == 0).sum() >= 8]
sids = sorted(np.random.default_rng(cfg['seed']).choice(alls, size=max(6, len(alls) // 10), replace=False))
G = np.array([brands.index(by_s[s].brand.iloc[0]) for s in sids])
own = np.array([float(by_s[s].own.iloc[0]) for s in sids], np.float32)
arr = hbnn.build(d, sids, lambda s, a: a.outlier == 0, T, z)
iBP = COLS.index('lnBP')
out = {'series': list(sids)}
for kind in ['M2', 'M3']:
    f = hbnn.fit(kind, arr, G, own, len(brands), steps=cfg.get('svi_steps', 8000))
    ps = hbnn.posterior(f, n=1000)
    model = hbnn.m2_model if kind == 'M2' else hbnn.m3_model
    mc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=1000, num_samples=1000, progress_bar=False)
    mc.run(jax.random.PRNGKey(1), **f['kw'])
    sites = ['b', 'sigma'] + (['rho'] if kind == 'M3' else [])
    pn = Predictive(model, posterior_samples=mc.get_samples(), return_sites=sites)(jax.random.PRNGKey(2), **f['kw'])
    pn = {k: np.asarray(v) for k, v in pn.items()}
    div = int(np.asarray(mc.get_extra_fields()['diverging']).sum())
    bs, bn = ps['b'][:, :, iBP], pn['b'][:, :, iBP]
    ws = np.quantile(bs, .95, 0) - np.quantile(bs, .05, 0)
    wn = np.quantile(bn, .95, 0) - np.quantile(bn, .05, 0)
    out[kind] = {'beta_median_svi': np.median(bs, 0).round(3).tolist(),
                 'beta_median_nuts': np.median(bn, 0).round(3).tolist(),
                 'max_abs_median_diff': float(np.abs(np.median(bs, 0) - np.median(bn, 0)).max()),
                 'width_ratio_svi_over_nuts': (ws / wn).round(3).tolist(),
                 'median_width_ratio': float(np.median(ws / wn)),
                 'sigma_ratio_median': float(np.median(np.median(ps['sigma'], 0) / np.median(pn['sigma'], 0))),
                 'nuts_divergences': div}
    if kind == 'M3':
        out[kind]['rho_svi'] = float(np.median(ps['rho'])); out[kind]['rho_nuts'] = float(np.median(pn['rho']))
    print(kind, json.dumps(out[kind]), flush=True)
json.dump(out, open('reports/tables/svi_vs_nuts.json', 'w'), indent=1)
