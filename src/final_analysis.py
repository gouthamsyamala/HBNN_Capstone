"""Stage 7 (final): full-sample M2 / M3 fits and the analyses that answer RQ2-RQ4.

M3 here is the two-state model (local level + dynamic elasticity); lift baselines and the decomposition
include the smoothed level, so promotion effects are measured against the week's adapted baseline.
Outputs (reports/tables/final_analysis.json unless noted)
  elasticities      static (M2) and dynamic (M3) base-price elasticities by series and brand
                    -> elasticities_final.csv (series-level estimates only, no volumes)
  rq2_paths         RTS-smoothed elasticity paths; brand means (fig14); driver regression of
                    quarterly elasticity changes on distribution, competitor promotion and sentiment
  rq3               event lift against the M3 baseline (promotion terms set to zero); Kruskal-Wallis,
                    ANCOVA on depth; posterior contrasts of mechanic coefficients; post-promotion dip
  rq4               unconstrained media coefficient (H0_4 test), constrained contributions and value per
                    dollar under four media priors (sensitivity), Granger-type test of sales -> media
  importance        permutation feature importance of M3 on the 52-week backtest (4 refits) and
                    contribution-variance shares from the full fit (fig16)
  decomposition     volume due to base, promotion and media by brand (fig17)
  scenarios         +5% list price and promotion what-if examples with 90% CrI
"""
import json
import sys
import time

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
sys.path.insert(0, 'src')
import hbnn  # noqa: E402
from models import COLS  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
STEPS, NDRAW = cfg.get('svi_steps', 8000), 400
T_ = 'reports/tables/'
FIG = 'reports/figures/'
t00 = time.time()
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = hbnn.add_drivers(d)
z_all = hbnn.attributes(d[d.archived == 0])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
T = int(d.t.max()) + 1
brands = sorted(d.brand.unique())
by_s = {s: g.sort_values('t') for s, g in d.groupby('series_id')}
sids = [s for s in sorted(by_s) if (by_s[s].outlier == 0).sum() >= 8]
G = np.array([brands.index(by_s[s].brand.iloc[0]) for s in sids])
own = np.array([float(by_s[s].own.iloc[0]) for s in sids], np.float32)
arr = hbnn.build(d, sids, lambda s, a: a.outlier == 0, T, z_all)
week_of_t = d.groupby('t').week.first()
vol_w = np.array([by_s[s].Volume_Sales.sum() for s in sids])
ci = lambda a: [float(np.quantile(a, .05)), float(np.median(a)), float(np.quantile(a, .95))]  # noqa: E731
out = {}
iBP, iPR = COLS.index('lnBP'), COLS.index('lnPR')
MECH = {'TPR': 's_tpr', 'FO': 's_fo', 'DO': 's_do', 'F&D': 's_fd'}

# literature-based media prior (Sethuraman et al., 2011: mean short-term advertising elasticity ~0.12;
# prior median of the media coefficient set to 0.05 with a wide raw-scale sd of 1)
inv_sp = lambda v: float(np.log(np.expm1(v)))  # noqa: E731
hbnn.MEDIA.update(mode='pos', loc=inv_sp(0.05), scale=1.0)


def fit_post(kind, a=arr, seed=640):
    f = hbnn.fit(kind, a, G, own, len(brands), steps=STEPS, seed=seed)
    return f, hbnn.posterior(f, n=NDRAW, seed=seed)


f2, p2 = fit_post('M2')
f3, p3 = fit_post('M3')
print(f'fits done {time.time() - t00:.0f}s', flush=True)
paths, levels = hbnn.smoothed_states(f3, p3, arr)                 # draws x J x T each
lev_med = np.median(levels, 0)
np.savez_compressed(T_ + 'm3_paths.npz', paths_q=np.quantile(paths, [.05, .5, .95], axis=0), sids=np.array(sids))


def recent(j, k=13):
    """Current elasticity = average of the smoothed path over the series' last k observed weeks (per draw);
    more stable than the single end-point of the smoothed path."""
    tt = np.where(arr['M_all'][j])[0][-k:]
    return paths[:, j, tt].mean(1)
cv = d.groupby('series_id').PriceBase_Volume.agg(lambda x: x.std() / x.mean())

# ----------------------------------------------------------------------------- elasticities
last_t = np.array([np.where(arr['M_all'][j])[0].max() for j in range(len(sids))])
first_t = np.array([np.where(arr['M_all'][j])[0].min() for j in range(len(sids))])
m1 = pd.read_csv(T_ + 'elasticities_full.csv').set_index('series_id')
rows = []
for j, s in enumerate(sids):
    pj = paths[:, j]
    rows.append({'series_id': s, 'brand': brands[G[j]], 'basepx_cv': float(cv[s]),
                 'M1_lnBP': float(m1.loc[s, 'M1_lnBP']) if s in m1.index else np.nan,
                 'M2_lnBP': float(np.median(p2['b'][:, j, iBP])),
                 'M2_lnBP_lo': float(np.quantile(p2['b'][:, j, iBP], .05)),
                 'M2_lnBP_hi': float(np.quantile(p2['b'][:, j, iBP], .95)),
                 'M2_lnPR': float(np.median(p2['b'][:, j, iPR])),
                 'M3_lnBP_longrun': float(np.median(p3['b'][:, j, iBP])),
                 'M3_lnBP_first': float(np.median(pj[:, first_t[j]])),
                 'M3_lnBP_current': float(np.median(recent(j))),
                 'M3_lnBP_current_lo': float(np.quantile(recent(j), .05)),
                 'M3_lnBP_current_hi': float(np.quantile(recent(j), .95)),
                 'M3_change_p_more_elastic': float((recent(j) < pj[:, np.where(arr['M_all'][j])[0][:13]].mean(1)).mean()),
                 's_eta': float(np.median(p3['s_eta'][:, j])),
                 'M3_path_range': float(np.ptp(np.median(pj[:, first_t[j]:last_t[j] + 1], 0)))})
el = pd.DataFrame(rows)
el.to_csv(T_ + 'elasticities_final.csv', index=False)
CUR = np.stack([recent(j) for j in range(len(sids))], 1)          # draws x J


def wmean(x, idx):
    w = vol_w[idx] / vol_w[idx].sum()
    return (x[:, idx] * w).sum(1)


out['elasticities'] = {
    'category_M2_lnBP': ci(wmean(p2['b'][:, :, iBP], np.arange(len(sids)))),
    'category_M2_lnPR': ci(wmean(p2['b'][:, :, iPR], np.arange(len(sids)))),
    'category_M3_lnBP_longrun': ci(wmean(p3['b'][:, :, iBP], np.arange(len(sids)))),
    'category_M3_lnPR': ci(wmean(p3['b'][:, :, iPR], np.arange(len(sids)))),
    'category_M3_lnBP_current': ci(wmean(CUR, np.arange(len(sids)))),
    'brand': {b: {'M2_lnBP': ci(wmean(p2['b'][:, :, iBP], np.where(G == g)[0])),
                  'M3_lnBP_current': ci(wmean(CUR, np.where(G == g)[0])),
                  'M3_lnPR': ci(wmean(p3['b'][:, :, iPR], np.where(G == g)[0]))}
              for g, b in enumerate(brands)},
    'wrong_sign_M2_lnBP': float((el.M2_lnBP > 0).mean()), 'n_series': int(len(el)),
    'rho': ci(p3['rho']), 'rho_level': ci(p3['rho_l_']),
    's_eta_median': float(np.median(p3['s_eta'])), 's_eta_series_range': [float(np.median(p3['s_eta'], 0).min()), float(np.median(p3['s_eta'], 0).max())],
    'level_sd_median': float(np.median(np.std(lev_med, 1))),
    'sigma_median_M2': float(np.median(p2['sigma'])),
    'sigma_median_M3': float(np.median(p3['sigma'])),
    'series_credible_change_95': int(((el.M3_change_p_more_elastic > .95) | (el.M3_change_p_more_elastic < .05)).sum()),
    'series_more_elastic_95': int((el.M3_change_p_more_elastic > .95).sum()),
    'series_less_elastic_95': int((el.M3_change_p_more_elastic < .05).sum()),
    'median_path_range': float(el.M3_path_range.median()),
    'path_range_lowcv': float(el[el.basepx_cv < .02].M3_path_range.median()),
    'path_range_highcv': float(el[el.basepx_cv >= .02].M3_path_range.median()),
    's_eta_lowcv_median': float(el[el.basepx_cv < .02].s_eta.median()),
    's_eta_highcv_median': float(el[el.basepx_cv >= .02].s_eta.median()),
    'n_lowcv': int((el.basepx_cv < .02).sum()),
    'series_large_s_eta_gt2': el[el.s_eta > 2][['series_id', 'basepx_cv', 's_eta']].round(4).to_dict('records')}
r2 = []
mu = np.einsum('jtp,sjp->sjt', arr['X'], p3['b'])
fitted = np.median(mu + arr['X'][None, :, :, iBP] * (paths - p3['b'][:, :, iBP][:, :, None]) + levels, 0)
for j in range(len(sids)):
    m = arr['M_tr'][j]
    y = arr['Y'][j, m]
    r2.append(1 - ((y - fitted[j, m]) ** 2).sum() / ((y - y.mean()) ** 2).sum())
out['elasticities']['insample_r2_median_M3'] = float(np.median(r2))

# ----------------------------------------------------------------------------- RQ2 paths and drivers
wk = week_of_t.reindex(range(T)).values
fig, ax = plt.subplots(figsize=(7.5, 3.4))
COLB = dict(zip(['Own Brand A', 'Own Brand B', 'Competitor C', 'Competitor D', 'Competitor E', 'Competitor F',
                 'Private Label'], ['#1f4e79', '#4f81bd', '#c0504d', '#9bbb59', '#8064a2', '#f79646', '#7f7f7f']))
brand_paths = {}
for g, b in enumerate(brands):
    idx = np.where(G == g)[0]
    w = (arr['M_all'][idx] * vol_w[idx][:, None])
    wsum = w.sum(0)
    bp = (paths[:, idx, :] * w[None]).sum(1) / np.where(wsum > 0, wsum, 1)
    ok = wsum > 0
    brand_paths[b] = bp
    lo, md, hi = np.quantile(bp[:, ok], [.05, .5, .95], axis=0)
    ax.plot(wk[ok], md, color=COLB[b], lw=1.4 if b.startswith('Own') else 1, label=b)
    if b.startswith('Own'):
        ax.fill_between(wk[ok], lo, hi, color=COLB[b], alpha=.15)
ax.axvline(pd.Timestamp('2026-03-08'), color='k', ls=':', lw=.7)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax.set_ylabel('Base-price elasticity (smoothed)')
ax.legend(frameon=False, fontsize=7, ncol=4, loc='lower left')
fig.tight_layout()
fig.savefig(FIG + 'fig14_dynamic_elasticity.png', dpi=200)
plt.close()
out['rq2_brand_paths_quarterly'] = {
    b: [(str(pd.Timestamp(wk[t]).date()), *[round(float(x), 3) for x in np.quantile(v[:, t], [.05, .5, .95])])
        for t in list(range(12, T, 13)) + [T - 1]] for b, v in brand_paths.items()}

# driver regression: quarterly change in each series' smoothed elasticity on quarterly changes in drivers
med_path = np.median(paths, 0)
reg = []
for j, s in enumerate(sids):
    a = by_s[s].set_index('t')
    for t0 in range(first_t[j], last_t[j] - 12, 13):
        t1 = t0 + 13
        if t1 > last_t[j] or t0 not in a.index or t1 not in a.index:
            continue
        reg.append({'series_id': s, 'dE': med_path[j, t1] - med_path[j, t0],
                    'd_lnTPD': a.loc[t1, 'lnTPD'] - a.loc[t0, 'lnTPD'],
                    'd_comp_depth': a.loc[t1 - 3:t1, 'comp_depth'].mean() - a.loc[t0 - 3:t0, 'comp_depth'].mean(),
                    'd_csi': a.loc[t1, 'csi'] - a.loc[t0, 'csi'],
                    'd_lnBP': a.loc[t1, 'lnBP'] - a.loc[t0, 'lnBP']})
reg = pd.DataFrame(reg)
Xr = sm.add_constant((reg[['d_lnTPD', 'd_comp_depth', 'd_csi', 'd_lnBP']] - reg[['d_lnTPD', 'd_comp_depth', 'd_csi', 'd_lnBP']].mean())
                     / reg[['d_lnTPD', 'd_comp_depth', 'd_csi', 'd_lnBP']].std())
ols = sm.OLS(reg.dE, Xr).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(reg.series_id)[0]})
out['rq2_drivers'] = {'n_obs': int(len(reg)), 'n_series': int(reg.series_id.nunique()), 'r2': float(ols.rsquared),
                      'coef_per_sd': {k: [float(ols.params[k]), float(ols.bse[k]), float(ols.pvalues[k])]
                                      for k in ols.params.index}}
print(f'RQ2 done {time.time() - t00:.0f}s', flush=True)

# ----------------------------------------------------------------------------- RQ3
PIDX = [COLS.index(c) for c in ['lnPR', 's_tpr', 's_fo', 's_do', 's_fd']]
bmed = np.median(p3['b'], 0)
smed = np.median(p3['sigma'], 0)
eta_med = med_path - bmed[:, iBP][:, None]
ev, resid_rows = [], []
for j, s in enumerate(sids):
    a = by_s[s]
    tt = a.t.values
    Xj = arr['X'][j, tt]
    full = Xj @ bmed[j] + Xj[:, iBP] * eta_med[j, tt] + lev_med[j, tt]
    Xb = Xj.copy(); Xb[:, PIDX] = 0
    base = Xb @ bmed[j] + Xb[:, iBP] * eta_med[j, tt] + lev_med[j, tt]
    resid = arr['Y'][j, tt] - full
    pe = a.promo_event.values
    for k in range(len(tt)):
        if pe[k] != 'None':
            ev.append({'series_id': s, 'brand': brands[G[j]], 'own': own[j], 'mechanic': pe[k],
                       'depth': 1 - a.promo_ratio.values[k],
                       'loglift': float(arr['Y'][j, tt[k]] - base[k]) if a.outlier.values[k] == 0 else np.nan})
        if k > 0 and pe[k - 1] != 'None' and pe[k] == 'None' and tt[k] == tt[k - 1] + 1 and a.outlier.values[k] == 0:
            resid_rows.append(resid[k])
ev = pd.DataFrame(ev).dropna()
ev['lift'] = np.expm1(ev.loglift)
M = ['TPR', 'FO', 'DO', 'F&D']
grp = [ev.loc[ev.mechanic == m, 'loglift'].values for m in M]
kw = stats.kruskal(*grp)
Xa = np.column_stack([np.ones(len(ev))] + [(ev.mechanic == m).astype(float) for m in M[1:]] + [ev.depth])
Xr = np.column_stack([np.ones(len(ev)), ev.depth])
yv = ev.loglift.values
bA, rss_f = np.linalg.lstsq(Xa, yv, rcond=None)[:2]
rss_f = float(((yv - Xa @ bA) ** 2).sum())
rss_r = float(((yv - Xr @ np.linalg.lstsq(Xr, yv, rcond=None)[0]) ** 2).sum())
df2 = len(yv) - Xa.shape[1]
F = ((rss_r - rss_f) / 3) / (rss_f / df2)
mech_draw = {m: wmean(p3['b'][:, :, COLS.index(c)], np.arange(len(sids))) for m, c in MECH.items()}
tab = ev.groupby('mechanic').agg(n=('lift', 'size'), median_depth=('depth', 'median'),
                                 median_lift=('lift', 'median'), mean_loglift=('loglift', 'mean'),
                                 sd_loglift=('loglift', 'std')).loc[M]
own_tab = ev.groupby(['own', 'mechanic']).lift.median().unstack(0)
rr = np.array(resid_rows)
out['rq3'] = {'table': tab.round(4).reset_index().to_dict('records'), 'n_events': int(len(ev)),
              'kruskal_H': float(kw.statistic), 'kruskal_p': float(kw.pvalue),
              'levene_p': float(stats.levene(*grp, center='median').pvalue),
              'ancova_F': float(F), 'ancova_df': [3, int(df2)], 'ancova_p': float(stats.f.sf(F, 3, df2)),
              'partial_eta2': float((rss_r - rss_f) / rss_r),
              'ancova_coef_vs_TPR': dict(zip(['FO', 'DO', 'F&D', 'depth'], np.round(bA[1:], 4).tolist())),
              'posterior_lift_full_support': {m: [float(np.expm1(x)) for x in ci(v)] for m, v in mech_draw.items()},
              'posterior_contrasts': {f'P({a}>{b})': float((mech_draw[a] > mech_draw[b]).mean())
                                      for a in M for b in M if a != b},
              'promo_price_elasticity_M3': ci(wmean(p3['b'][:, :, iPR], np.arange(len(sids)))),
              'median_lift_own_vs_comp': {m: {'competitor': float(own_tab.loc[m, 0.0]), 'own': float(own_tab.loc[m, 1.0])}
                                          for m in M},
              'share_negative_lift': float((ev.lift < 0).mean()),
              'post_promo_dip': {'n_weeks': int(len(rr)), 'mean_resid': float(rr.mean()),
                                 't_p_two_sided': float(stats.ttest_1samp(rr, 0).pvalue),
                                 'mean_resid_pct': float(np.expm1(rr.mean()))}}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), gridspec_kw={'width_ratios': [1.1, 1]})
bp_ = axes[0].boxplot([np.expm1(g) * 100 for g in grp], tick_labels=M, showfliers=False, patch_artist=True,
                      medianprops={'color': 'k'})
for p_, c in zip(bp_['boxes'], ['#4f81bd', '#c0504d', '#9bbb59', '#8064a2']):
    p_.set_facecolor(c); p_.set_alpha(.7)
axes[0].axhline(0, color='k', lw=.6)
axes[0].set_ylabel('Event lift vs. M3 baseline (%)')
axes[0].set_title('Event-level lift, including discount', loc='left', fontsize=9)
for i, m in enumerate(M):
    lo, md, hi = out['rq3']['posterior_lift_full_support'][m]
    axes[1].errorbar(md * 100, i, xerr=[[100 * (md - lo)], [100 * (hi - md)]], fmt='o', color='#1f4e79', capsize=3)
    axes[1].text(hi * 100 + 1.5, i, f'{md * 100:.0f}%', va='center', fontsize=8)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(M); axes[1].invert_yaxis()
axes[1].axvline(0, color='k', lw=.6)
axes[1].set_xlabel('Lift at full support, price held (%)')
axes[1].set_title('M3 category posterior, 90% CrI', loc='left', fontsize=9)
fig.tight_layout()
fig.savefig(FIG + 'fig18_promo_lift_final.png', dpi=200)
plt.close()
print(f'RQ3 done {time.time() - t00:.0f}s', flush=True)

# ----------------------------------------------------------------------------- RQ4
iA = COLS.index('lnA')


def media_summary(post, eta_paths=None):
    res = {}
    for b in ['Own Brand A', 'Own Brand B']:
        g = brands.index(b)
        idx = np.where(G == g)[0]
        lam = wmean(post['b'][:, :, iA], idx)
        vol, val = np.zeros(post['b'].shape[0]), np.zeros(post['b'].shape[0])
        for j in idx:
            a = by_s[sids[j]]
            lj = post['b'][:, j, iA][:, None]
            inc = a.Volume_Sales.values[None] * (1 - np.exp(-lj * a.lnA.values[None]))
            vol += inc.sum(1)
            val += (inc * a.Price_per_Volume.values[None]).sum(1)
        spend = d[d.brand == b].groupby('week').Total_BrandMedia_Spends.first().sum()
        tot = d[d.brand == b].Volume_Sales.sum()
        res[b] = {'lambda': ci(lam), 'P_lambda_gt0': float((lam > 0).mean()),
                  'contribution_share': ci(vol / tot), 'value_per_dollar': ci(val / spend)}
    return res


rq4 = {'prior_literature_median_0.05': media_summary(p3)}
hbnn.MEDIA.update(mode='free', loc=0.0, scale=1.0)
_, pf = fit_post('M3', seed=641)
rq4['unconstrained'] = media_summary(pf)
for lab, v in [('prior_median_0.01', 0.01), ('prior_median_0.12', 0.12)]:
    hbnn.MEDIA.update(mode='pos', loc=inv_sp(v), scale=1.0)
    _, ps = fit_post('M3', seed=642)
    rq4[lab] = media_summary(ps)
hbnn.MEDIA.update(mode='pos', loc=0.0, scale=3.0)
_, pw = fit_post('M3', seed=643)
rq4['prior_weak_default'] = media_summary(pw)
hbnn.MEDIA.update(mode='pos', loc=inv_sp(0.05), scale=1.0)
gr = {}
for b in ['Own Brand A', 'Own Brand B']:
    sub = d[d.brand == b]
    w = sub.groupby('t').agg(vol=('Volume_Sales', 'sum'), media=('Total_BrandMedia_Spends', 'first')).sort_index()
    w = w.reindex(range(w.index.min(), w.index.max() + 1)).interpolate()
    x = np.column_stack([np.log(w.media), np.log(w.vol)])
    r = grangercausalitytests(x, maxlag=[4])[4][0]['ssr_ftest']
    gr[b] = {'F': float(r[0]), 'p': float(r[1]), 'df': [int(r[3]), int(r[2])]}
rq4['granger_sales_to_media_lag4'] = gr
out['rq4'] = rq4
print(f'RQ4 done {time.time() - t00:.0f}s', flush=True)

# ----------------------------------------------------------------------------- decomposition
dec = []
for j, s in enumerate(sids):
    a = by_s[s]
    tt = a.t.values
    Xj = arr['X'][j, tt]
    lv = Xj @ bmed[j] + Xj[:, iBP] * eta_med[j, tt]
    nop = Xj.copy(); nop[:, PIDX] = 0
    nom = Xj.copy(); nom[:, iA] = 0
    V = a.Volume_Sales.values
    vp = V * (1 - np.exp((nop @ bmed[j] + nop[:, iBP] * eta_med[j, tt]) - lv))
    vm = V * (1 - np.exp(-bmed[j, iA] * a.lnA.values))        # media switched off (uncentred adstock = 0)
    dec.append({'brand': brands[G[j]], 'volume': V.sum(), 'promo_incr': vp.sum(), 'media_incr': vm.sum()})
dec = pd.DataFrame(dec).groupby('brand').sum()
dec['promo_share'] = dec.promo_incr / dec.volume
dec['media_share'] = dec.media_incr / dec.volume
dec['base_share'] = 1 - dec.promo_share - dec.media_share
out['decomposition'] = dec[['promo_share', 'media_share', 'base_share']].round(4).to_dict('index')
cat = dec[['volume', 'promo_incr', 'media_incr']].sum()
out['decomposition_category'] = {'promo_share': float(cat.promo_incr / cat.volume),
                                 'media_share': float(cat.media_incr / cat.volume)}
fig, ax = plt.subplots(figsize=(7.5, 2.9))
o = dec.sort_values('promo_share')
ax.barh(o.index, o.base_share * 100, color='#bfbfbf', label='Base (price, distribution, season)')
ax.barh(o.index, o.promo_share * 100, left=o.base_share * 100, color='#4f81bd', label='Promotion (incremental)')
ax.barh(o.index, o.media_share * 100, left=(o.base_share + o.promo_share) * 100, color='#c0504d',
        label='Media (incremental)')
for i, (b, r_) in enumerate(o.iterrows()):
    ax.text(101, i, f'{r_.promo_share * 100:.0f}% promo', va='center', fontsize=7.5)
ax.set_xlim(0, 118)
ax.set_xlabel('Share of brand volume (%), Jan 2024 – Aug 2026')
ax.legend(frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(.45, -.45), ncol=3)
fig.tight_layout()
fig.savefig(FIG + 'fig17_decomposition.png', dpi=200)
plt.close()

# ----------------------------------------------------------------------------- feature importance
GROUPS = {'Base price': ['lnBP'], 'Promotional price ratio': ['lnPR'],
          'Mechanic support (TPR, FO, DO, F&D)': ['s_tpr', 's_fo', 's_do', 's_fd'],
          'Distribution': ['lnTPD'], 'Competitor price': ['lnCP'], 'Media': ['lnA'],
          'Seasonality and holidays': ['sin1', 'cos1', 'sin2', 'cos2', 'hol_thanks', 'hol_xmas', 'hol_easter'],
          'Trend and post-break': ['trend', 'post_break']}
rng = np.random.default_rng(cfg['seed'])
imp = {k: [] for k in GROUPS}
base_err = []
H = cfg['horizon']
for o in cfg['backtest_origins']:
    ao = hbnn.build(d, sids, lambda s, a, o=o: (a.t < o) & (a.outlier == 0), T, z_all)
    keep = [j for j in range(len(sids)) if ao['M_tr'][j].sum() >= 8 and ao['M_all'][j, o:o + H].any()]
    ao = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == (len(sids),) else v) for k, v in ao.items()}
    Gk, ownk = G[keep], own[keep]
    fo = hbnn.fit('M3', ao, Gk, ownk, len(brands), steps=STEPS, seed=cfg['seed'] + o)
    po = hbnn.posterior(fo, n=200, seed=o)
    test = np.zeros_like(ao['M_all']); test[:, o:o + H] = ao['M_all'][:, o:o + H]

    def err(a_):
        dr, _, _ = hbnn.predictive(fo, po, a_, ao['M_tr'], seed=o)
        yhat = np.median(np.exp(dr + ao['ybar'][None, :, None]), 0)
        y = np.exp(ao['Y'] + ao['ybar'][:, None])
        return np.abs(y - yhat)[test].sum() / y[test].sum() * 100
    e0 = err(ao)
    base_err.append(e0)
    for gname, cols in GROUPS.items():
        ci_ = [COLS.index(c) for c in cols]
        reps = []
        for _ in range(10):
            a2 = dict(ao); X2 = ao['X'].copy()
            for j in range(len(keep)):
                tt = np.where(test[j])[0]
                perm = rng.permutation(len(tt))
                X2[j][np.ix_(tt, ci_)] = ao['X'][j][np.ix_(tt[perm], ci_)]
            a2['X'] = X2
            reps.append(err(a2) - e0)
        imp[gname].append(np.mean(reps))
    print(f'  importance origin {o} done {time.time() - t00:.0f}s', flush=True)
contrib = {}
Xall = arr['X']
for gname, cols in GROUPS.items():
    ci_ = [COLS.index(c) for c in cols]
    c = np.einsum('jtp,jp->jt', Xall[:, :, ci_], bmed[:, ci_])
    if 'lnBP' in cols:
        c = c + Xall[:, :, iBP] * eta_med
    v = [np.var(c[j, arr['M_tr'][j]]) for j in range(len(sids))]
    contrib[gname] = float(np.median(v))
tot_v = sum(contrib.values())
out['importance'] = {'method': 'permutation within series over the 13-week test window, 10 repeats, '
                               'M3 refitted at each of the 4 backtest origins; increase in pooled WMAPE (pp)',
                     'baseline_pooled_WMAPE': [float(x) for x in base_err],
                     'permutation_increase_pp': {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))]
                                                 for k, v in imp.items()},
                     'contribution_variance_share': {k: v / tot_v for k, v in contrib.items()}}
o_ = pd.Series({k: np.mean(v) for k, v in imp.items()}).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 3.0))
ax.barh(o_.index, o_.values, color='#1f4e79')
for i, v in enumerate(o_.values):
    ax.text(v + .05, i, f'{v:.2f}', va='center', fontsize=8)
ax.set_xlabel('Increase in pooled test WMAPE when permuted (percentage points)')
fig.tight_layout()
fig.savefig(FIG + 'fig16_feature_importance.png', dpi=200)
plt.close()

# ----------------------------------------------------------------------------- scenarios
sc = {}
for b in ['Own Brand A', 'Own Brand B']:
    idx = np.where(G == brands.index(b))[0]
    cur = CUR[:, idx]                                                 # draws x series
    w = vol_w[idx] / vol_w[idx].sum()
    vf = (1.05 ** cur * w).sum(1)
    sc[b] = {'plus5pct_list_price': {'volume_change': ci(vf - 1), 'value_change': ci(1.05 * vf - 1)}}
    fd = p3['b'][:, idx, COLS.index('s_fd')]
    tp = p3['b'][:, idx, COLS.index('s_tpr')]
    pr = p3['b'][:, idx, iPR]
    v_tpr = ((np.exp(pr * np.log(0.8) + tp)) * w).sum(1)
    v_fd = ((np.exp(pr * np.log(0.8) + fd)) * w).sum(1)
    sc[b]['20pct_off_TPR_only'] = {'volume_lift': ci(v_tpr - 1)}
    sc[b]['20pct_off_feature_and_display'] = {'volume_lift': ci(v_fd - 1)}
out['scenarios'] = sc
json.dump(out, open(T_ + 'final_analysis.json', 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ['elasticities', 'rq2_drivers']}, indent=1, default=float)[:3000])
print('total', round(time.time() - t00), 's')
