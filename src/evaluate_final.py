"""Stage 6 (final): accuracy of M0-M3, RQ1 tests and the RQ2 leave-future-out comparison.

RQ1  primary  : one-sided Wilcoxon signed-rank test on series-level WMAPE, M3 vs the better benchmark
     secondary: one-sided paired t-test; Diebold-Mariano test (HAC variance, Harvey et al. correction)
                on the weekly cross-series mean loss differential; per-series DM tests.
RQ2  for each series, the leave-future-out ELPD difference M3 - M2 summed over the 52 backtest weeks,
     using one-step-ahead densities (M3's filter is updated with each observed week, parameters
     re-estimated every 13 weeks); proportion of series with a positive difference, Wilson 95% CI and
     one-sided exact binomial test of H0: p <= 0.50.
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
T = 'reports/tables/'
fc = pd.concat([pd.read_csv(T + 'backtest_forecasts.csv'), pd.read_csv(T + 'backtest_forecasts_hbnn.csv')])
fc['yhat'] = fc.yhat.clip(upper=1e12)
d = pd.read_csv('data/processed/panel_clean.csv', keep_default_na=False, na_values=[''])
nat = set(d[(d.sparse_flag == 1) & (d.archived == 0)].series_id.unique())
cv = d[d.archived == 0].groupby('series_id').PriceBase_Volume.agg(lambda x: x.std() / x.mean())
MODELS = ['M0', 'M1', 'M2', 'M3']
fc = fc[fc.model.isin(MODELS)]          # M3L (level-only) is used for RQ2 only
fc['group'] = np.where(fc.series_id.isin(nat), 'sparse_natural',
                       np.where(fc.scenario == 'trunc', 'sparse_truncated', 'rich'))
# only keep series-weeks that every model forecast, so comparisons are paired
key = ['origin', 'scenario', 'series_id', 't']
cnt = fc.groupby(key).model.nunique()
fc = fc.set_index(key).loc[cnt[cnt == len(MODELS)].index].reset_index()


def metrics(g):
    e = np.abs(g.y - g.yhat)
    return pd.Series({'WMAPE': e.sum() / g.y.sum() * 100,
                      'RMSE_log': np.sqrt(np.mean((np.log(g.y) - np.log(g.yhat)) ** 2)),
                      'MASE': np.mean(e / g.scale),
                      'Coverage90': np.mean((g.y >= g.lo) & (g.y <= g.hi)),
                      'sCRPS': np.nanmean(np.where(np.isfinite(g.crps), g.crps, np.nan)) / g.y.mean(),
                      'weeks': len(g)})


ser = fc.groupby(['group', 'series_id', 'model']).apply(metrics, include_groups=False).reset_index()
ser['is_sparse'] = ser.group.str.startswith('sparse')
ser.to_csv(T + 'series_metrics_final.csv', index=False)
out = {}
for name, sub in [('rich', ser[~ser.is_sparse]), ('sparse', ser[ser.is_sparse])]:
    s = sub.groupby('model')[['WMAPE', 'RMSE_log', 'MASE', 'Coverage90', 'sCRPS']].median()
    s['mean_WMAPE'] = sub.groupby('model').WMAPE.mean()
    s['n_series'] = sub.groupby('model').size()
    out[f'summary_{name}'] = s.round(4).reset_index().to_dict('records')
bench = ser[ser.is_sparse].groupby('model').WMAPE.median()[['M0', 'M1']].idxmin()
out['best_benchmark_sparse'] = bench


def paired(sub, a, b):
    """Improvement of model a over model b (positive = a better), in WMAPE percentage points."""
    w = sub.pivot_table(index='series_id', columns='model', values='WMAPE').dropna()
    diff = w[b] - w[a]
    return {'n': int(len(diff)), 'median_WMAPE_a': float(w[a].median()), 'median_WMAPE_b': float(w[b].median()),
            'median_improvement_pp': float(diff.median()), 'mean_improvement_pp': float(diff.mean()),
            'trimmed_mean_10pct': float(stats.trim_mean(diff, .1)), 'share_a_better': float((diff > 0).mean()),
            'wilcoxon_p_one_sided': float(stats.wilcoxon(diff, alternative='greater').pvalue),
            'wilcoxon_p_two_sided': float(stats.wilcoxon(diff).pvalue),
            't': float(stats.ttest_1samp(diff, 0).statistic),
            't_p_one_sided': float(stats.ttest_1samp(diff, 0, alternative='greater').pvalue),
            'shapiro_p': float(stats.shapiro(diff).pvalue),
            'cohens_d': float(diff.mean() / diff.std(ddof=1)),
            'matched_rank_biserial': float(rank_biserial(diff))}


def rank_biserial(x):
    x = x[x != 0]
    r = stats.rankdata(np.abs(x))
    return (r[x > 0].sum() - r[x < 0].sum()) / r.sum()


def dm_test(loss_diff, h=13):
    """Diebold-Mariano with Newey-West variance (lag h-1) and Harvey-Leybourne-Newbold correction."""
    x = np.asarray(loss_diff, float)
    n = len(x)
    xm = x - x.mean()
    gam = [np.dot(xm[k:], xm[:n - k]) / n for k in range(h)]
    v = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))
    dm = x.mean() / np.sqrt(v / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * corr
    return float(stat), float(stats.t.sf(stat, n - 1))   # one-sided: positive = second model better


for grp, sub in [('sparse', ser[ser.is_sparse]), ('rich', ser[~ser.is_sparse])]:
    out[f'rq1_{grp}'] = {f'{a}_vs_{b}': paired(sub, a, b)
                         for a, b in [('M3', bench), ('M2', bench), ('M3', 'M2'), ('M1', 'M0')]}
    # Diebold-Mariano on scaled absolute errors: weekly mean loss differential across series
    f = fc[fc.group.str.startswith('sparse')] if grp == 'sparse' else fc[fc.group == 'rich']
    f = f.assign(loss=np.abs(f.y - f.yhat) / f.groupby('series_id').y.transform('mean'))
    pv = f.pivot_table(index=['series_id', 'scenario', 'origin', 't'], columns='model', values='loss').dropna()
    dmres = {}
    for a, b in [('M3', bench), ('M2', bench), ('M3', 'M2')]:
        ld = (pv[b] - pv[a]).groupby(level='t').mean().sort_index()
        st, p = dm_test(ld.values)
        per = []
        for s, g in (pv[b] - pv[a]).groupby(level='series_id'):
            g = g.groupby(level='t').mean().sort_index()
            if len(g) >= 26:
                per.append(dm_test(g.values))
        per = np.array(per)
        dmres[f'{a}_vs_{b}'] = {'dm_stat': st, 'p_one_sided': p, 'weeks': int(len(ld)),
                                'series_tested': int(len(per)),
                                'series_a_sig_better': int((per[:, 1] < .05).sum()),
                                'series_b_sig_better': int((stats.t.sf(-per[:, 0], 51) < .05).sum())}
    out[f'dm_{grp}'] = dmres

# ----------------------------------------------------------------------------------- RQ2
dens = pd.read_csv(T + 'lfo_density.csv')
dn = dens[dens.scenario == 'full'].pivot_table(index=['series_id', 'origin', 't'], columns='model',
                                               values=['lpd_1step', 'lpd_multi'])


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def elpd_compare(a, b):
    diff = dn[('lpd_1step', a)] - dn[('lpd_1step', b)]
    e1 = diff.groupby(level='series_id').sum()
    se = diff.groupby(level='series_id').agg(lambda x: np.sqrt(len(x)) * x.std(ddof=1))
    em = (dn[('lpd_multi', a)] - dn[('lpd_multi', b)]).groupby(level='series_id').sum()
    r = pd.DataFrame({'elpd_diff_1step': e1, 'se_1step': se, 'elpd_diff_multi': em})
    r['basepx_cv'] = r.index.map(cv)
    r['low_variation'] = r.basepx_cv < .02
    k, n = int((r.elpd_diff_1step > 0).sum()), int(len(r))
    res = {'n_series': n, 'n_first_better': k, 'share': k / n, 'wilson95': wilson(k, n),
           'binom_p_one_sided': float(stats.binomtest(k, n, .5, alternative='greater').pvalue),
           'total_elpd_diff_1step': float(e1.sum()), 'total_se': float(np.sqrt((se ** 2).sum())),
           'median_elpd_diff_1step': float(e1.median()),
           'n_clearly_better_2se': int((r.elpd_diff_1step > 2 * r.se_1step).sum()),
           'n_clearly_worse_2se': int((r.elpd_diff_1step < -2 * r.se_1step).sum()),
           'multi_step_share_first_better': float((em > 0).mean()),
           'low_variation': {'n': int(r.low_variation.sum()),
                             'share_better': float((r[r.low_variation].elpd_diff_1step > 0).mean())},
           'adequate_variation': {'n': int((~r.low_variation).sum()),
                                  'share_better': float((r[~r.low_variation].elpd_diff_1step > 0).mean())}}
    return r, res


rq2, out['rq2'] = elpd_compare('M3', 'M2')                 # pre-registered comparison
rq2b, out['rq2_vs_level_only'] = elpd_compare('M3', 'M3L')   # isolates elasticity dynamics
_, out['rq2_level_only_vs_M2'] = elpd_compare('M3L', 'M2')
rq2.join(rq2b[['elpd_diff_1step', 'se_1step']], rsuffix='_vs_M3L').to_csv(T + 'rq2_elpd_by_series.csv')
k, n = out['rq2']['n_first_better'], out['rq2']['n_series']
json.dump(out, open(T + 'final_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# ----------------------------------------------------------------------------------- figures
COL = {'M0': '#c0504d', 'M1': '#7f7f7f', 'M2': '#4f81bd', 'M3': '#1f4e79'}
LAB = {'M0': 'M0 OLS', 'M1': 'M1 HB', 'M2': 'M2 HBNN', 'M3': 'M3 HBNN-KF'}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
for ax, (grp, lab) in zip(axes, [('rich', 'Data-rich series (full history)'),
                                 ('sparse', 'Data-sparse series (26-week histories)')]):
    sub = ser[ser.is_sparse] if grp == 'sparse' else ser[~ser.is_sparse]
    w = sub.pivot_table(index='series_id', columns='model', values='WMAPE').dropna()
    bp = ax.boxplot([w[m] for m in MODELS], positions=range(4), widths=.5, showfliers=False, patch_artist=True,
                    medianprops={'color': 'k'})
    for p_, m in zip(bp['boxes'], MODELS):
        p_.set_facecolor(COL[m]); p_.set_alpha(.75)
    for i, m in enumerate(MODELS):
        ax.text(i, w[m].quantile(.75) * 1.08, f'{w[m].median():.1f}', ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[m] for m in MODELS], fontsize=8)
    ax.set_title(f'{lab}, n = {len(w)}', loc='left', fontsize=8.5)
axes[0].set_ylabel('Series WMAPE, % (log scale)')
fig.tight_layout()
fig.savefig('reports/figures/fig13_backtest_wmape_all.png', dpi=200)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
for ax, (r, lab, key) in zip(axes, [(rq2, 'M3 − M2 (dynamic vs. static, planned test)', 'rq2'),
                                    (rq2b, 'M3 − M3L (isolates elasticity dynamics)', 'rq2_vs_level_only')]):
    o = r.sort_values('elpd_diff_1step').reset_index()
    c = np.where(o.low_variation, '#bfbfbf', np.where(o.elpd_diff_1step > 0, '#1f4e79', '#c0504d'))
    ax.bar(range(len(o)), o.elpd_diff_1step.clip(-40, 40), color=c, width=.8)
    ax.axhline(0, color='k', lw=.6)
    ax.set_xlabel('Series (sorted)')
    ax.set_title(f"{lab}\n{out[key]['n_first_better']} of {out[key]['n_series']} series favour M3", loc='left', fontsize=8)
    ax.set_ylim(-42, 42)
axes[0].set_ylabel('ELPD difference (one-step, 52 weeks)')
fig.text(.01, .01, 'Grey = base-price CV < 2%. Bars clipped at ±40.', fontsize=7)
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig('reports/figures/fig15_rq2_elpd.png', dpi=200)
plt.close()
