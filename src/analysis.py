"""Stage 6-7 (preliminary): full-sample M0/M1 fits, elasticity distribution, rolling-window elasticity
paths (RQ2 preview), event-level promotional lift by mechanic (RQ3), and media effects (RQ4)."""
import json
import pickle
import sys

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 10})
sys.path.insert(0, 'src')
from models import design, fit_m0, fit_m1, COLS  # noqa: E402

cfg = yaml.safe_load(open('configs/config.yaml'))
FIG = 'reports/figures/'
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = d[(d.archived == 0) & (d.low_acv_week == 0)].copy()
brands = sorted(d.brand.unique())
out = {}
ci = lambda a: [float(np.quantile(a, .05)), float(np.median(a)), float(np.quantile(a, .95))]  # noqa: E731


def build(sub):
    series, meta = [], []
    for s, a in sub.groupby('series_id'):
        a = a[a.outlier == 0]
        if len(a) < 8:
            continue
        X, means = design(a)
        series.append((X, a.lnV.values))
        meta.append((s, brands.index(a.brand.iloc[0]), means))
    return series, meta


# ---------------- full-sample fit
series, meta = build(d)
post = fit_m1(series, [m[1] for m in meta], n_iter=cfg['gibbs_iter'], burn=cfg['gibbs_burn'], seed=cfg['seed'])
pickle.dump({'post': post, 'meta': meta}, open('reports/tables/m1_full_posterior.pkl', 'wb'))
iBP, iPR = COLS.index('lnBP'), COLS.index('lnPR')
coef = []
for j, (s, g, _) in enumerate(meta):
    m0 = fit_m0(*series[j])
    B = post['B'][:, j]
    coef.append({'series_id': s, 'brand': brands[g], 'n': len(series[j][1]),
                 'M0_lnBP': m0['beta'][iBP], 'M0_lnPR': m0['beta'][iPR],
                 'M1_lnBP': np.median(B[:, iBP]), 'M1_lnBP_lo': np.quantile(B[:, iBP], .05),
                 'M1_lnBP_hi': np.quantile(B[:, iBP], .95), 'M1_lnPR': np.median(B[:, iPR])})
coef = pd.DataFrame(coef)
coef.to_csv('reports/tables/elasticities_full.csv', index=False)
out['wrong_sign_full'] = {'n': int(len(coef)),
                          'M0_lnBP_pos': float((coef.M0_lnBP > 0).mean()),
                          'M1_lnBP_pos': float((coef.M1_lnBP > 0).mean()),
                          'M0_lnPR_pos': float((coef.M0_lnPR > 0).mean()),
                          'M1_lnPR_pos': float((coef.M1_lnPR > 0).mean())}
out['category_coef'] = {c: ci(post['mu0'][:, k]) for k, c in enumerate(COLS)}
out['brand_lnBP'] = {brands[g]: ci(post['mu_g'][:, g, iBP]) for g in range(len(brands))}
out['brand_lnPR'] = {brands[g]: ci(post['mu_g'][:, g, iPR]) for g in range(len(brands))}
out['sigma_median'] = float(np.median(np.sqrt(post['s2'])))
# in-sample fit
r2 = []
for j, (X, y) in enumerate(series):
    yh = X @ np.median(post['B'][:, j], 0)
    r2.append(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
out['insample_r2_median'] = float(np.median(r2))

# Figure 9: elasticity shrinkage M0 vs M1
fig, ax = plt.subplots(figsize=(7.5, 3.4))
o = coef.sort_values('M1_lnBP').reset_index(drop=True)
ax.scatter(o.index, o.M0_lnBP.clip(-8, 4), s=14, color='#c0504d', label='M0 (series OLS)', zorder=3)
ax.errorbar(o.index, o.M1_lnBP, yerr=[o.M1_lnBP - o.M1_lnBP_lo, o.M1_lnBP_hi - o.M1_lnBP], fmt='o', ms=3.5,
            color='#1f4e79', ecolor='#9fb6cd', lw=1, label='M1 (hierarchical Bayes, 90% CrI)', zorder=4)
ax.axhline(0, color='k', lw=.6)
ax.set_ylim(-8.5, 4.5)
ax.set_xlabel('Series (sorted by M1 estimate)')
ax.set_ylabel('Base-price elasticity')
ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(FIG + 'fig9_elasticity_shrinkage.png', dpi=200)
plt.close()

# ---------------- RQ2 preview: rolling 52-week windows
ends = [52, 65, 78, 91, 104, 117, 130, 139]
paths, ser_paths = {}, []
for e in ends:
    sub = d[(d.t >= e - 52) & (d.t < e)]
    sr, mt = build(sub)
    pw = fit_m1(sr, [m[1] for m in mt], n_iter=1500, burn=700, seed=cfg['seed'] + e)
    wk = d[d.t == e - 1].week.iloc[0]
    for g in range(len(brands)):
        paths.setdefault(brands[g], []).append((wk, *ci(pw['mu_g'][:, g, iBP])))
    for j, (s, g, _) in enumerate(mt):
        ser_paths.append({'end_t': e, 'week': wk, 'series_id': s, 'brand': brands[g],
                          'draws': pw['B'][:, j, iBP]})
sp = pd.DataFrame(ser_paths)
first, last = sp[sp.end_t == ends[0]].set_index('series_id'), sp[sp.end_t == ends[-1]].set_index('series_id')
common = first.index.intersection(last.index)
chg = []
for s in common:
    a, b = first.loc[s, 'draws'], last.loc[s, 'draws']
    m = min(len(a), len(b))
    diff = b[:m] - a[:m]
    chg.append({'series_id': s, 'brand': first.loc[s, 'brand'], 'first': float(np.median(a)),
                'last': float(np.median(b)), 'diff': float(np.median(diff)),
                'p_more_elastic': float((diff < 0).mean())})
chg = pd.DataFrame(chg)
cv = d.groupby('series_id').PriceBase_Volume.agg(lambda x: x.std() / x.mean())
chg['basepx_cv'] = chg.series_id.map(cv)
chg.to_csv('reports/tables/rq2_window_change.csv', index=False)
sig = (chg.p_more_elastic > .95) | (chg.p_more_elastic < .05)
k, n = int(sig.sum()), int(len(chg))
z = 1.96
ph = k / n
den = 1 + z * z / n
cen = (ph + z * z / (2 * n)) / den
half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
out['rq2_preview'] = {'n_series': n, 'n_changed_90': k, 'share': ph, 'wilson95': [cen - half, cen + half],
                      'n_more_elastic': int((chg.p_more_elastic > .95).sum()),
                      'n_less_elastic': int((chg.p_more_elastic < .05).sum()),
                      'n_lowcv_lt2pct': int((chg.basepx_cv < .02).sum()),
                      'median_abs_change': float(chg['diff'].abs().median())}
out['rq2_brand_paths'] = {b: [(str(w.date()), round(lo, 3), round(md, 3), round(hi, 3)) for w, lo, md, hi in v]
                          for b, v in paths.items()}
fig, ax = plt.subplots(figsize=(7.5, 3.3))
COL = dict(zip(['Own Brand A', 'Own Brand B', 'Competitor C', 'Competitor D', 'Competitor E', 'Competitor F',
                'Private Label'], ['#1f4e79', '#4f81bd', '#c0504d', '#9bbb59', '#8064a2', '#f79646', '#7f7f7f']))
for b in ['Own Brand A', 'Own Brand B', 'Competitor C', 'Competitor D', 'Private Label']:
    v = np.array([(x[1], x[2], x[3]) for x in paths[b]])
    wks = [x[0] for x in paths[b]]
    ax.plot(wks, v[:, 1], marker='o', ms=3, color=COL[b], label=b)
    if b.startswith('Own'):
        ax.fill_between(wks, v[:, 0], v[:, 2], color=COL[b], alpha=.12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax.set_ylabel('Brand-mean base-price elasticity')
ax.set_xlabel('End of 52-week estimation window')
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower left')
fig.tight_layout()
fig.savefig(FIG + 'fig10_rolling_elasticity.png', dpi=200)
plt.close()

# ---------------- RQ3: event-level lift (actual vs. model counterfactual without promotion)
P = ['lnPR', 's_tpr', 's_fo', 's_do', 's_fd']
pidx = [COLS.index(c) for c in P]
ev = []
for j, (s, g, means) in enumerate(meta):
    a = d[(d.series_id == s) & (d.promo_event != 'None')]
    if a.empty:
        continue
    X, _ = design(a, means)
    Xb = X.copy()
    Xb[:, pidx] = 0.0
    bmed = np.median(post['B'][:, j], 0)
    s2 = np.median(post['s2'][:, j])
    base = np.exp(Xb @ bmed + s2 / 2)
    ev.append(pd.DataFrame({'series_id': s, 'brand': brands[g], 'mechanic': a.promo_event.values,
                            'depth': 1 - a.promo_ratio.values, 'actual': a.Volume_Sales.values, 'base': base,
                            'price': a.Price_per_Volume.values, 'own': a.own.values}))
ev = pd.concat(ev, ignore_index=True)
ev['lift'] = ev.actual / ev.base - 1
ev['loglift'] = np.log(ev.actual / ev.base)
ev['incr'] = ev.actual - ev.base
ev.to_csv('reports/tables/rq3_events.csv', index=False)
M = ['TPR', 'FO', 'DO', 'F&D']
grp = [ev.loc[ev.mechanic == m, 'loglift'].values for m in M]
kw = stats.kruskal(*grp)
lev = stats.levene(*grp, center='median')
an = stats.f_oneway(*grp)
# ANCOVA: loglift ~ mechanic + depth
Xa = np.column_stack([np.ones(len(ev))] + [(ev.mechanic == m).astype(float) for m in M[1:]] + [ev.depth])
Xr = np.column_stack([np.ones(len(ev)), ev.depth])
yv = ev.loglift.values
rss_f = ((yv - Xa @ np.linalg.lstsq(Xa, yv, rcond=None)[0]) ** 2).sum()
rss_r = ((yv - Xr @ np.linalg.lstsq(Xr, yv, rcond=None)[0]) ** 2).sum()
df1, df2 = 3, len(yv) - Xa.shape[1]
F = ((rss_r - rss_f) / df1) / (rss_f / df2)
bA = np.linalg.lstsq(Xa, yv, rcond=None)[0]
eta2 = (rss_r - rss_f) / ((yv - yv.mean()) ** 2).sum()
tab = ev.groupby('mechanic').agg(n=('lift', 'size'), median_lift=('lift', 'median'), mean_loglift=('loglift', 'mean'),
                                 sd_loglift=('loglift', 'std'), median_depth=('depth', 'median'),
                                 incr_share=('incr', lambda x: x.sum()))
tab['incr_share'] = tab.incr_share / ev.loc[ev.incr > -1e18, 'actual'].groupby(ev.mechanic).sum()
tab = tab.loc[M]
out['rq3'] = {'table': tab.round(3).reset_index().to_dict('records'),
              'kruskal_H': float(kw.statistic), 'kruskal_p': float(kw.pvalue),
              'anova_F': float(an.statistic), 'anova_p': float(an.pvalue),
              'levene_p': float(lev.pvalue), 'ancova_F_mech': float(F), 'ancova_df': [df1, int(df2)],
              'ancova_p': float(1 - stats.f.cdf(F, df1, df2)), 'partial_eta2_approx': float(eta2),
              'ancova_coef_vs_TPR': dict(zip(['FO', 'DO', 'F&D', 'depth'], np.round(bA[1:], 3).tolist())),
              'n_events': int(len(ev)), 'share_negative_lift': float((ev.lift < 0).mean())}
# posterior contrasts on category-level support coefficients (lift at 100% ACV support, holding price)
th = {m: post['mu0'][:, COLS.index(c)] for m, c in zip(M, ['s_tpr', 's_fo', 's_do', 's_fd'])}
out['rq3_posterior'] = {m: [float(np.expm1(x)) for x in ci(v)] for m, v in th.items()}
out['rq3_contrasts'] = {f'P({a}>{b})': float((th[a] > th[b]).mean()) for a in M for b in M if a != b}
# Figure 11
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), gridspec_kw={'width_ratios': [1.1, 1]})
bp = axes[0].boxplot([np.expm1(g) * 100 for g in grp], tick_labels=M, showfliers=False, patch_artist=True,
                     medianprops={'color': 'k'})
for patch, c in zip(bp['boxes'], ['#4f81bd', '#c0504d', '#9bbb59', '#8064a2']):
    patch.set_facecolor(c); patch.set_alpha(.7)
axes[0].axhline(0, color='k', lw=.6)
axes[0].set_ylabel('Event lift vs. model baseline (%)')
axes[0].set_title('Event-level lift (incl. discount)', loc='left', fontsize=9)
for i, m in enumerate(M):
    lo, md, hi = out['rq3_posterior'][m]
    axes[1].errorbar(md * 100, i, xerr=[[100 * (md - lo)], [100 * (hi - md)]], fmt='o', color='#1f4e79', capsize=3)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(M)
axes[1].invert_yaxis()
axes[1].axvline(0, color='k', lw=.6)
axes[1].set_xlabel('Lift at full support, price held (%)')
axes[1].set_title('Category posterior (90% CrI)', loc='left', fontsize=9)
fig.tight_layout()
fig.savefig(FIG + 'fig11_promo_lift.png', dpi=200)
plt.close()

# ---------------- RQ4: media
iA = COLS.index('lnA')
rq4 = {}
for b in ['Own Brand A', 'Own Brand B']:
    g = brands.index(b)
    lam = post['mu_g'][:, g, iA]
    nd = post['B'].shape[0]
    contrib_vol, contrib_val = np.zeros(nd), np.zeros(nd)
    for j, (s, gg, means) in enumerate(meta):
        if gg != g:
            continue
        a = d[d.series_id == s]
        lamj = post['B'][:, j, iA][:, None]                                  # draws x 1
        base = a.Volume_Sales.values[None, :] * np.exp(-lamj * a.lnA.values[None, :])  # media set to zero
        inc = a.Volume_Sales.values[None, :] - base
        contrib_vol += inc.sum(1)
        contrib_val += (inc * a.Price_per_Volume.values[None, :]).sum(1)
    spend = d[d.brand == b].groupby('week').Total_BrandMedia_Spends.first().sum()
    tot = d[d.brand == b].Volume_Sales.sum()
    rq4[b] = {'lambda_brand': ci(lam), 'P_lambda_pos': float((lam > 0).mean()),
              'contrib_share_vol': ci(contrib_vol / tot),
              'roi_two_retailers': ci(contrib_val / spend),     # spend levels are confidential: ratio only
              'mean_lnA': float(d[d.brand == b].lnA.mean())}
out['rq4'] = rq4
json.dump(out, open('reports/tables/analysis_results.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'rq2_brand_paths'}, indent=1, default=str))
