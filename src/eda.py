"""Stage 3: exploratory data analysis. Writes figures to reports/figures and stats to reports/tables."""
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 10, 'figure.dpi': 110})
d = pd.read_csv('data/processed/panel_clean.csv', parse_dates=['week'], keep_default_na=False, na_values=[''])
d = d[d.archived == 0].copy()
FIG = 'reports/figures/'


def datefmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.tick_params(axis='x', labelsize=7.5)

st = {}
BR = ['Own Brand A', 'Own Brand B', 'Competitor C', 'Competitor D', 'Competitor E', 'Competitor F', 'Private Label']
COL = dict(zip(BR, ['#1f4e79', '#4f81bd', '#c0504d', '#9bbb59', '#8064a2', '#f79646', '#7f7f7f']))
MCOL = {'TPR': '#4f81bd', 'FO': '#c0504d', 'DO': '#9bbb59', 'F&D': '#8064a2', 'None': '#bfbfbf'}

# ---- Figure 1: weekly category volume and promoted share by retailer
fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
for ax, mk in zip(axes, ['Retailer 1', 'Retailer 2']):
    x = d[d.market == mk].groupby('week')[['Volume_Sales', 'vol_tpr', 'vol_fo', 'vol_do', 'vol_fd']].sum()
    promo = x[['vol_tpr', 'vol_fo', 'vol_do', 'vol_fd']].sum(axis=1)
    ax.fill_between(x.index, 0, (x.Volume_Sales - promo) / 1e3, color='#bfbfbf', label='Non-promoted')
    ax.fill_between(x.index, (x.Volume_Sales - promo) / 1e3, x.Volume_Sales / 1e3, color='#4f81bd', label='Promoted')
    ax.set_title(mk, loc='left')
    ax.set_ylabel('Volume (000 lb)')
    st[f'promo_share_{mk}'] = float(promo.sum() / x.Volume_Sales.sum())
    st[f'weekly_vol_cv_{mk}'] = float(x.Volume_Sales.std() / x.Volume_Sales.mean())
    st[f'weekly_vol_maxmin_{mk}'] = float(x.Volume_Sales.max() / x.Volume_Sales.median())
axes[0].legend(frameon=False, ncol=2, loc='upper right')
datefmt(axes[1])
fig.tight_layout()
fig.savefig(FIG + 'fig1_category_volume.png', dpi=200)
plt.close()

# ---- Figure 2: brand share vs average price per lb
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)
shares = {}
for ax, mk in zip(axes, ['Retailer 1', 'Retailer 2']):
    x = d[d.market == mk].groupby('brand').agg(v=('Volume_Sales', 'sum'), s=('Dollar_Sales', 'sum'))
    x['share'] = x.v / x.v.sum() * 100
    x['ppl'] = x.s / x.v
    shares[mk] = x[['share', 'ppl']].round(3).to_dict()
    for b, r in x.iterrows():
        ax.scatter(r.ppl, r.share, s=40 + r.share * 25, color=COL[b], alpha=.85)
        off = {'Private Label': (14, -8), 'Own Brand A': (6, -12), 'Own Brand B': (8, 2),
               'Competitor E': (-10, -14), 'Competitor C': (12, 6)}.get(b, (5, 3))
        ax.annotate(b.replace('Competitor', 'Comp.').replace('Own Brand', 'Own'), (r.ppl, r.share),
                    xytext=off, textcoords='offset points', fontsize=7.5)
    ax.set_title(mk, loc='left')
    ax.set_xlabel('Average price per lb ($)')
axes[0].set_ylabel('Volume share (%)')
st['brand_share_price'] = shares
fig.tight_layout()
fig.savefig(FIG + 'fig2_share_price.png', dpi=200)
plt.close()

# ---- Figure 3: base-price restatement (balanced panel, fixed volume weights, Jan 2024 = 100)
full = d.groupby('series_id').week.nunique()
bal = d[d.series_id.isin(full[full == 139].index)]
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
bp = {}
for ax, mk in zip(axes, ['Retailer 1', 'Retailer 2']):
    x = bal[bal.market == mk]
    w = x.groupby('series_id').Volume_Sales.sum()
    res = {}
    for col, lab, c, ls in [('PriceBase_Volume_raw', 'Base price (as delivered)', '#c0504d', '-'),
                            ('PriceBase_Volume', 'Base price (restated)', '#1f4e79', '-'),
                            ('Price_per_Volume', 'Average price paid', '#7f7f7f', '--')]:
        p = x.pivot_table(index='week', columns='series_id', values=col)
        idx = (p / p.iloc[:4].mean()).mul(w / w.sum(), axis=1).sum(axis=1) * 100
        idx = idx.rolling(4, min_periods=1).mean() if col == 'Price_per_Volume' else idx
        ax.plot(idx.index, idx.values, color=c, ls=ls, lw=1.4, label=lab)
        res[col] = [float(idx.loc[:'2026-03-01'].iloc[-4:].mean()), float(idx.iloc[-4:].mean())]
    bp[mk] = res
    ax.axvline(pd.Timestamp('2026-03-08'), color='k', lw=.6, ls=':')
    ax.set_title(mk, loc='left')
    datefmt(ax)
axes[0].set_ylabel('Price index (Jan 2024 = 100)')
axes[0].legend(frameon=False, fontsize=7, loc='lower left')
st['price_index_prebreak_end'] = bp
fig.tight_layout()
fig.savefig(FIG + 'fig3_base_price_index.png', dpi=200)
plt.close()

# ---- Figure 4: promoted volume share by mechanic and brand
x = d.groupby('brand')[['Volume_Sales', 'vol_tpr', 'vol_fo', 'vol_do', 'vol_fd']].sum()
p = x[['vol_tpr', 'vol_fo', 'vol_do', 'vol_fd']].div(x.Volume_Sales, axis=0) * 100
p.columns = ['TPR', 'FO', 'DO', 'F&D']
p = p.loc[BR]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
left = np.zeros(len(p))
for m in p.columns:
    ax.barh(p.index, p[m], left=left, color=MCOL[m], label=m)
    left += p[m].values
for i, v in enumerate(left):
    ax.text(v + .5, i, f'{v:.0f}%', va='center', fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Share of brand volume sold on promotion (%)')
ax.legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
ax.set_xlim(0, 75)
st['promo_share_by_brand'] = p.round(1).to_dict()
fig.tight_layout()
fig.savefig(FIG + 'fig4_promo_mix.png', dpi=200)
plt.close()

# ---- Figure 5: discount depth vs volume uplift over non-promoted median
d['depth'] = 1 - d.promo_ratio
base = d[d.promo_event == 'None'].groupby('series_id').apply(
    lambda g: np.median(g.lnV - g.lnTPD), include_groups=False)
d['uplift'] = np.exp(d.lnV - d.lnTPD - d.series_id.map(base)) - 1
e = d[(d.promo_event != 'None') & d.uplift.notna()]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for m in ['TPR', 'FO', 'DO', 'F&D']:
    z = e[e.promo_event == m]
    ax.scatter(z.depth * 100, 1 + z.uplift, s=7, alpha=.45, color=MCOL[m], label=f'{m} (n={len(z)})')
ax.set_yscale('log')
ax.set_yticks([0.5, 1, 2, 4, 8])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%gx'))
ax.minorticks_off()
ax.axhline(1, color='k', lw=.6)
ax.set_xlabel('Average discount depth in week (%)')
ax.set_ylabel('Volume / non-promoted median (log scale)')
ax.legend(frameon=False, fontsize=7.5)
st['raw_uplift_median_by_mech'] = e.groupby('promo_event').uplift.median().round(3).to_dict()
st['depth_median_by_mech'] = e.groupby('promo_event').depth.median().round(3).to_dict()
st['corr_depth_lnuplift'] = float(np.corrcoef(e.depth, np.log1p(e.uplift.clip(lower=-.99)))[0, 1])
fig.tight_layout()
fig.savefig(FIG + 'fig5_depth_uplift.png', dpi=200)
plt.close()

# ---- Figure 6: series history length vs distribution (sparsity)
sr = d.groupby('series_id').agg(n=('week', 'nunique'), acv=('acv', 'mean'), brand=('brand', 'first'),
                                cv=('PriceBase_Volume', lambda s: s.std() / s.mean()))
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for b in BR:
    z = sr[sr.brand == b]
    ax.scatter(z.n, z.acv, color=COL[b], s=30, label=b, alpha=.85)
ax.axvline(52, ls='--', color='k', lw=.7)
ax.axhline(20, ls='--', color='k', lw=.7)
ax.text(22, 3, 'sparse: < 52 weeks or < 20% ACV', fontsize=7.5)
ax.set_ylim(0, 105)
ax.set_xlabel('Weeks of sales history')
ax.set_ylabel('Mean weighted distribution (%ACV)')
ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower right')
fig.tight_layout()
fig.savefig(FIG + 'fig6_series_sparsity.png', dpi=200)
plt.close()
st['series_n_weeks'] = sr.n.describe().round(1).to_dict()
st['basepx_cv_lt2pct'] = int((sr.cv < .02).sum())
st['basepx_cv_median'] = float(sr.cv.median())
st['basepx_cv_quartiles'] = sr.cv.quantile([.25, .5, .75]).round(4).to_list()

# ---- Figure 7: own-brand media
own = d[d.own == 1]
ch = {'Video (TV/OLV/Audio)': ['Linear_TV', 'Digital_TV', 'OLV', 'Audio'],
      'Social/Influencer/Partner': ['Paid_Social', 'Influencer', 'Partnership'],
      'Search/Display/DOOH': ['Paid_Search', 'OLA', 'DOOH']}
fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.4), sharex=True)
media_st = {}
# spend is confidential: plot an index (Own Brand A's average weekly spend = 100), never dollar amounts
ref = own[own.brand == 'Own Brand A'].groupby('week').Total_BrandMedia_Spends.first().mean() / 100
for ax, b in zip(axes, ['Own Brand A', 'Own Brand B']):
    m = own[own.brand == b].groupby('week').first()
    bottom = np.zeros(len(m))
    for (k, cols), c in zip(ch.items(), ['#1f4e79', '#9bbb59', '#f79646']):
        v = m[cols].sum(axis=1).values / ref
        ax.bar(m.index, v, bottom=bottom, width=6, color=c, label=k)
        bottom += v
    ax.set_title(b + ' (national brand media)', loc='left')
    ax.set_ylabel('Spend index')
    tot = m.Total_BrandMedia_Spends
    media_st[b] = {'relative_total_vs_A': float(tot.sum() / (ref * 100 * len(m))), 'weeks_zero': int((tot == 0).sum()),
                   'cv': float(tot.std() / tot.mean()),
                   'share_video': float(m[ch['Video (TV/OLV/Audio)']].sum().sum() / tot.sum())}
axes[0].legend(frameon=False, fontsize=7.5, ncol=3)
datefmt(axes[1])
st['media'] = media_st
fig.tight_layout()
fig.savefig(FIG + 'fig7_media.png', dpi=200)
plt.close()

# ---- Figure 8: seasonality (monthly index of distribution-adjusted volume)
d['month'] = d.week.dt.month
x = d.groupby(['series_id', 'month']).apply(lambda g: np.mean(g.lnV - g.lnTPD), include_groups=False).unstack()
x = np.exp(x.sub(x.mean(axis=1), axis=0)).mean() * 100
fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.bar(x.index, x.values, color=['#c0504d' if v > 104 else '#4f81bd' for v in x.values])
ax.axhline(100, color='k', lw=.6)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_ylabel('Seasonal index (avg = 100)')
ax.set_ylim(80, max(120, x.max() + 3))
st['season_index'] = x.round(1).to_dict()
fig.tight_layout()
fig.savefig(FIG + 'fig8_seasonality.png', dpi=200)
plt.close()

# ---- VIF of the demand drivers
X = d[['lnBP', 'lnPR', 's_tpr', 's_fo', 's_do', 's_fd', 'lnTPD', 'lnCP']].copy()
X = X - d.groupby('series_id')[X.columns].transform('mean')  # within-series variation
vif = {}
for c in X.columns:
    y = X[c].values
    Z = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
    r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    vif[c] = float(1 / (r.var() / y.var()))
st['vif_within'] = vif
st['corr_within'] = X.corr().round(2).to_dict()
st['csi_first_last'] = [float(d.csi.iloc[0]), float(d[d.csi_missing == 0].csi.iloc[-1])]
csi_w = d.groupby('week').csi.first()
st['csi_range'] = [float(csi_w.min()), float(csi_w.max())]
json.dump(st, open('reports/tables/eda_stats.json', 'w'), indent=1, default=str)
print(json.dumps(st, indent=1, default=str))
