"""Stage 6: series-level accuracy metrics and preliminary RQ1 tests (M1 vs M0)."""
import json

import numpy as np
import pandas as pd
from scipy import stats

fc = pd.read_csv('reports/tables/backtest_forecasts.csv')
fc['yhat'] = fc.yhat.clip(upper=1e12)
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
nat = set(d[(d.sparse_flag == 1) & (d.archived == 0)].series_id.unique())


def metrics(g):
    e = np.abs(g.y - g.yhat)
    return pd.Series({
        'WMAPE': e.sum() / g.y.sum() * 100,
        'RMSE_log': np.sqrt(np.mean((np.log(g.y) - np.log(g.yhat)) ** 2)),
        'MASE': np.mean(e / g.scale),
        'Coverage90': np.mean((g.y >= g.lo) & (g.y <= g.hi)),
        'sCRPS': np.nanmean(np.where(np.isfinite(g.crps), g.crps, np.nan)) / g.y.mean(),
        'weeks': len(g)})


fc['group'] = np.where(fc.series_id.isin(nat), 'sparse_natural',
                       np.where(fc.scenario == 'trunc', 'sparse_truncated', 'rich'))
ser = fc.groupby(['group', 'series_id', 'model']).apply(metrics, include_groups=False).reset_index()
ser.to_csv('reports/tables/series_metrics.csv', index=False)
ser['is_sparse'] = ser.group.str.startswith('sparse')

out = {}
summ = ser.groupby(['group', 'model'])[['WMAPE', 'RMSE_log', 'MASE', 'Coverage90', 'sCRPS']].median()
summ['n_series'] = ser.groupby(['group', 'model']).size()
out['median_by_group'] = summ.round(3).reset_index().to_dict('records')
# pooled sparse
sp = ser[ser.is_sparse]
summ2 = sp.groupby('model')[['WMAPE', 'RMSE_log', 'MASE', 'Coverage90', 'sCRPS']].agg(['mean', 'median'])
out['sparse_all'] = {m: {f'{c}_{s}': float(summ2.loc[m, (c, s)]) for c, s in summ2.columns} for m in summ2.index}

# paired test on sparse set
for name, sub in [('sparse', sp), ('rich', ser[~ser.is_sparse])]:
    w = sub.pivot_table(index='series_id', columns='model', values='WMAPE')
    w = w.dropna()
    diff = (w['M0'] - w['M1'])  # positive = M1 better (lower WMAPE), in percentage points
    t = stats.ttest_1samp(diff, 0, alternative='greater')
    wil = stats.wilcoxon(diff, alternative='greater')
    sh = stats.shapiro(diff)
    out[f'test_{name}'] = {
        'n': int(len(diff)), 'mean_improvement_pp': float(diff.mean()), 'median_improvement_pp': float(diff.median()),
        'sd': float(diff.std(ddof=1)), 'cohens_d': float(diff.mean() / diff.std(ddof=1)),
        't': float(t.statistic), 'p_t_one_sided': float(t.pvalue), 'wilcoxon_p': float(wil.pvalue),
        'shapiro_p': float(sh.pvalue), 'share_M1_better': float((diff > 0).mean()),
        'mean_WMAPE_M0': float(w['M0'].mean()), 'mean_WMAPE_M1': float(w['M1'].mean()),
        'trimmed_mean_improvement_pp_10pct': float(stats.trim_mean(diff, 0.1))}

# coefficient sign checks on truncated (26-week) fits at last origin
ct = pd.read_csv('reports/tables/coef_truncated_M1.csv')
out['wrong_sign_truncated'] = {
    'n': int(len(ct)),
    'M0_lnBP_pos': float((ct.M0_lnBP > 0).mean()), 'M1_lnBP_pos': float((ct.M1_lnBP > 0).mean()),
    'M0_lnPR_pos': float((ct.M0_lnPR > 0).mean()), 'M1_lnPR_pos': float((ct.M1_lnPR > 0).mean()),
    'M0_lnBP_iqr': ct.M0_lnBP.quantile([.25, .75]).round(2).tolist(),
    'M1_lnBP_iqr': ct.M1_lnBP.quantile([.25, .75]).round(2).tolist(),
    'M0_lnBP_extreme_abs_gt10': float((ct.M0_lnBP.abs() > 10).mean())}
# power achieved for the census of sparse series
n = out['test_sparse']['n']
nc = 0.30 * np.sqrt(n)
crit = stats.t.ppf(0.95, n - 1)
out['power_d030_one_sided'] = float(1 - stats.nct.cdf(crit, n - 1, nc))
json.dump(out, open('reports/tables/rq1_results.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

# Figure 12: series-level WMAPE by model and group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1))
for ax, (grp, lab) in zip(axes, [('rich', 'Data-rich series (full history)'),
                                 ('sparse', 'Data-sparse series (natural + 26-week truncation)')]):
    sub = ser[ser.is_sparse] if grp == 'sparse' else ser[~ser.is_sparse]
    w = sub.pivot_table(index='series_id', columns='model', values='WMAPE').dropna()
    for _, r in w.iterrows():
        ax.plot([0, 1], [r.M0, r.M1], color='#bfbfbf', lw=.6, zorder=1)
    ax.boxplot([w.M0, w.M1], positions=[0, 1], widths=.35, showfliers=False, medianprops={'color': 'k'})
    ax.set_yscale('log')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['M0 OLS', 'M1 HB'])
    ax.set_title(lab, loc='left', fontsize=8.5)
    ax.set_ylim(2, 1000)
axes[0].set_ylabel('Series WMAPE (%, log scale)')
fig.tight_layout()
fig.savefig('reports/figures/fig12_backtest_wmape.png', dpi=200)
