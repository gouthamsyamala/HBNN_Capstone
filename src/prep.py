"""Stages 1-2: validation, cleaning, masking and feature engineering.
Usage: python src/prep.py [path_to_datacube.xlsx]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open('configs/config.yaml'))
for _d in ('data/processed', 'reports/tables', 'reports/figures'):
    os.makedirs(_d, exist_ok=True)
raw = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else cfg['raw_file'], sheet_name='Data')
log = {'raw_rows': raw.shape[0], 'raw_cols': raw.shape[1]}
df = raw.copy()

# ---------- masking (confidentiality) ----------
# The raw-name -> masked-label mapping lives in configs/masking.yaml, which is git-ignored so
# that real brand and retailer names never enter the repository. See configs/masking.example.yaml.
try:
    MASK = yaml.safe_load(open(cfg.get('masking_file', 'configs/masking.yaml')))
except FileNotFoundError:
    sys.exit('configs/masking.yaml not found: copy configs/masking.example.yaml and fill in the raw names.')
BRAND, TOK, GEO = MASK['brand'], MASK['product_token'], MASK['geography']
for col, m in (('Brand', BRAND), ('Geography', GEO)):
    unmapped = sorted(set(df[col].unique()) - set(m))
    if unmapped:
        sys.exit(f'Unmapped {col} values in masking.yaml: {unmapped}')
df['brand'] = df['Brand'].map(BRAND)
df['own'] = df['brand'].str.startswith('Own Brand').astype(int)
df['market'] = df['Geography'].map(GEO)


def mask_name(s):
    t = s.split(' ')
    t[0] = TOK.get(t[0], t[0])
    return ' '.join(t).replace('_', '.')


df['sku'] = df['ProductName'].map(mask_name)
df['series_id'] = 'R' + df['market'].str[-1] + '-' + df['PPG_No'].astype(str).str.zfill(2)
df['week'] = pd.to_datetime(df['Week'])

# ---------- validation checks ----------
log['duplicates_key_week'] = int(df.duplicated(['Key', 'Week']).sum())
mech_sum = df[['Volume_Sales_Price_Reductions_Only', 'Volume_Sales_Feature_Only',
               'Volume_Sales_Display_Only', 'Volume_Sales_Feature_and_Display']].sum(axis=1)
log['mechanic_volume_exceeds_total'] = int((mech_sum > df['Volume_Sales'] * 1.001).sum())
log['value_volume_reconcile_maxdev'] = float(
    (df.Dollar_Sales / (df.Volume_Sales * df.Price_per_Volume) - 1).abs().max())
log['nonpositive_volume'] = int((df.Volume_Sales <= 0).sum())
log['missing_cells_total'] = int(raw.isna().sum().sum())

# ---------- cleaning ----------
disc = ['Discount_TPR', 'Discount_Feature_Only', 'Discount_Display_Only', 'Discount_Feature_and_Display']
log['discount_missing'] = int(df[disc].isna().sum().sum())
df[disc] = df[disc].fillna(0.0)  # missing only where the mechanic sold zero volume
log['discount_negative'] = int((df[disc] < 0).sum().sum())
df[disc] = df[disc].clip(lower=0)  # promoted price above base price -> treated as no discount
# base-price restatement: category base price drops ~33% in the week ending 2026-03-08 while the
# price actually paid is unchanged -> treated as a change in the provider's base-price method.
# Post-break base price is rescaled per series so that the ratio of paid price to base price in
# low-promotion weeks (<10% of volume on promotion) matches the pre-break ratio.
BRK = pd.Timestamp('2026-03-08')
df['week_dt'] = pd.to_datetime(df['Week'])
df['post_break'] = (df.week_dt >= BRK).astype(int)
cw = df.groupby('week_dt').apply(lambda g: np.average(g.PriceBase_Volume, weights=g.Volume_Sales),
                                 include_groups=False)
log['base_price_break_logchange'] = float(np.log(cw).diff().loc[BRK])
psh = df[['Volume_Sales_Price_Reductions_Only', 'Volume_Sales_Feature_Only', 'Volume_Sales_Display_Only',
          'Volume_Sales_Feature_and_Display']].sum(axis=1) / df.Volume_Sales
r = df.Price_per_Volume / df.PriceBase_Volume
lo = df[psh < 0.10].assign(r=r)
pre = lo[lo.post_break == 0].groupby('series_id').r.median()
post = lo[lo.post_break == 1].groupby('series_id').r.agg(['median', 'size'])
fac = (post['median'] / pre).where(post['size'] >= 4).dropna()
mk_of = df.groupby('series_id').market.first()
mk_med = fac.groupby(mk_of.reindex(fac.index)).median()
f_all = pd.Series({s: fac.get(s, mk_med[mk_of[s]]) for s in mk_of.index})
log['rebase_factor_median_by_market'] = mk_med.round(3).to_dict()
log['rebase_series_own_factor'] = int(len(fac))
df['PriceBase_Volume_raw'] = df.PriceBase_Volume
adj = df.series_id.map(f_all)
df.loc[df.post_break == 1, 'PriceBase_Volume'] = df.PriceBase_Volume * adj
cw2 = df.groupby('week_dt').apply(lambda g: np.average(g.PriceBase_Volume, weights=g.Volume_Sales),
                                  include_groups=False)
log['base_price_break_logchange_after_fix'] = float(np.log(cw2).diff().loc[BRK])
ratio = df.Price_per_Volume / df.PriceBase_Volume
log['price_ratio_above_1'] = int((ratio > 1.0).sum())
df['promo_ratio'] = ratio.clip(upper=1.0)

# consumer sentiment: populated only on own-brand rows and only up to 2026-03-01; zeros = missing
csi = df[df.Consumer_Sentiment_Index > 0].groupby('week').Consumer_Sentiment_Index.first()
log['csi_rows_zero'] = int((df.Consumer_Sentiment_Index == 0).sum())
allw = pd.Series(sorted(df.week.unique()))
csi = csi.reindex(allw.values)
log['csi_weeks_missing'] = int(csi.isna().sum())
df['csi_missing'] = df.week.map(csi.isna()).astype(int)
df['csi'] = df.week.map(csi.ffill())

# series filters (archived series are retained in the file, but not modelled)
ser = df.groupby('series_id').agg(n=('week', 'nunique'), acv=('ACV_Weighted_Distribution', 'mean'))
arch = ser[(ser.n < cfg['min_weeks_model']) | (ser.acv < cfg['min_mean_acv_model'])].index
log['series_total'] = int(len(ser))
log['series_archived'] = list(arch)
log['rows_archived'] = int(df.series_id.isin(arch).sum())
df['archived'] = df.series_id.isin(arch).astype(int)
df['low_acv_week'] = (df.ACV_Weighted_Distribution < cfg['low_acv_week']).astype(int)
log['low_acv_weeks_in_model_series'] = int(df.loc[df.archived == 0, 'low_acv_week'].sum())

# gaps inside each series' span
g = df.groupby('series_id').apply(
    lambda x: ((x.week.max() - x.week.min()).days // 7 + 1) - x.week.nunique(), include_groups=False)
log['internal_missing_weeks'] = int(g[~g.index.isin(arch)].sum())
log['series_with_gaps'] = int((g[~g.index.isin(arch)] > 0).sum())

# outliers: robust z-score of ln(volume / points of distribution) within series
df['lnV'] = np.log(df.Volume_Sales)


def rz(x):
    r = x.lnV - np.log(x.Total_Points_of_Distribution)
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    return (r - med) / (mad if mad > 0 else 1)


df['robust_z'] = df.groupby('series_id', group_keys=False).apply(rz, include_groups=False)
df['outlier'] = (df.robust_z.abs() > cfg['outlier_z']).astype(int)
log['outliers_flagged'] = int(df.loc[df.archived == 0, 'outlier'].sum())

# ---------- features ----------
df = df.sort_values(['series_id', 'week']).reset_index(drop=True)
df['t'] = ((df.week - df.week.min()).dt.days // 7).astype(int)
df['lnBP'] = np.log(df.PriceBase_Volume)
df['lnPR'] = np.log(df.promo_ratio)
for m, c in [('tpr', 'Price_Reductions_Only'), ('fo', 'Feature_Only'), ('do', 'Display_Only'),
             ('fd', 'Feature_and_Display')]:
    df['s_' + m] = df['Weighted_Weeks_v2_' + c]
    df['vol_' + m] = df['Volume_Sales_' + c]
df['lnTPD'] = np.log(df.Total_Points_of_Distribution)
df['acv'] = df.ACV_Weighted_Distribution

# competitor base-price index: fixed volume-share weights of other brands' PPGs in the same retailer
w = df.groupby(['market', 'brand', 'series_id']).Volume_Sales.sum().rename('w').reset_index()
piv = df.pivot_table(index='week', columns='series_id', values='PriceBase_Volume')
comp = {}
for mk in df.market.unique():
    for b in df.brand.unique():
        o = w[(w.market == mk) & (w.brand != b)]
        P = piv[o.series_id]
        W = P.notna() * o.set_index('series_id').w
        comp[(mk, b)] = (P.fillna(0) * W).sum(axis=1) / W.sum(axis=1)
df['comp_price'] = [comp[(m, b)][wk] for m, b, wk in zip(df.market, df.brand, df.week)]
df['lnCP'] = np.log(df.comp_price)

# media: brand-level national spend (replicated to every PPG) with geometric adstock
a = cfg['adstock_alpha']


def adstock(x):
    out, c = np.zeros(len(x)), 0.0
    for i, v in enumerate(x):
        c = v + a * c
        out[i] = c
    return out


bm = df[df.own == 1].groupby(['brand', 'week'])[['Total_BrandMedia_Spends']].first().reset_index()
bm['adstock_media'] = bm.groupby('brand').Total_BrandMedia_Spends.transform(lambda s: adstock(s.values))
df = df.merge(bm[['brand', 'week', 'adstock_media']], on=['brand', 'week'], how='left')
df['adstock_media'] = df.adstock_media.fillna(0)
df['lnA'] = np.log1p(df.adstock_media / 100000)  # concave saturation of adstock (in $100k)
df = df.sort_values(['series_id', 'week']).reset_index(drop=True)
df['adstock_retail'] = df.groupby('series_id').Total_RetailMedia.transform(lambda s: adstock(s.values))

# seasonality and holidays
doy = df.week.dt.dayofyear.values
for k in range(1, cfg['fourier_K'] + 1):
    df[f'sin{k}'] = np.sin(2 * np.pi * k * doy / 365.25)
    df[f'cos{k}'] = np.cos(2 * np.pi * k * doy / 365.25)


def week_has(dates):
    # week-ending dates: the week covers the 7 days up to and including week_ending
    return df.week.apply(lambda wk: any(0 <= (wk - pd.Timestamp(d)).days < 7 for d in dates)).astype(int)


df['hol_thanks'] = week_has(['2024-11-28', '2025-11-27'])
df['hol_xmas'] = week_has(['2024-12-25', '2025-12-25'])
df['hol_easter'] = week_has(['2024-03-31', '2025-04-20', '2026-04-05'])

# dominant promotional mechanic per promoted series-week (RQ3 grouping)
# merchandising hierarchy: F&D > FO / DO (larger support) > TPR only; support threshold 25% of ACV
th = 0.25
df['promo_event'] = np.select(
    [df.s_fd >= th, (df.s_fo >= th) & (df.s_fo >= df.s_do), df.s_do >= th, df.s_tpr >= th],
    ['F&D', 'FO', 'DO', 'TPR'], default='None')

# natural sparse flag
ser = df[df.archived == 0].groupby('series_id').agg(n=('week', 'nunique'), acv=('acv', 'mean'))
sp = ser[(ser.n < cfg['sparse_weeks']) | (ser.acv < cfg['sparse_acv'])].index
df['sparse_flag'] = df.series_id.isin(sp).astype(int)
log['series_modelled'] = int(len(ser))
log['series_sparse_natural'] = int(len(sp))
log['sparse_series'] = list(sp)
log['clean_rows_modelled'] = int((df.archived == 0).sum())
log['events_by_mechanic'] = df[df.archived == 0].promo_event.value_counts().to_dict()

keep = ['series_id', 'market', 'brand', 'own', 'sku', 'PackSize', 'week', 't', 'Volume_Sales', 'Dollar_Sales',
        'lnV', 'Price_per_Volume', 'PriceBase_Volume', 'promo_ratio', 'lnBP', 'lnPR', 's_tpr', 's_fo', 's_do',
        's_fd', 'vol_tpr', 'vol_fo', 'vol_do', 'vol_fd', 'Discount_TPR', 'Discount_Feature_Only',
        'Discount_Display_Only', 'Discount_Feature_and_Display', 'acv', 'Total_Points_of_Distribution', 'lnTPD',
        'Avg_Weekly_Items_per_Store_Selling', 'comp_price', 'lnCP', 'Total_BrandMedia_Spends',
        'Total_RetailMedia', 'adstock_media', 'adstock_retail', 'lnA', 'csi', 'csi_missing', 'post_break', 'PriceBase_Volume_raw', 'sin1', 'cos1',
        'sin2', 'cos2', 'hol_thanks', 'hol_xmas', 'hol_easter', 'promo_event', 'sparse_flag', 'archived',
        'low_acv_week', 'outlier', 'robust_z']
media_cols = ['Audio', 'Digital_TV', 'OLV', 'OLA', 'Paid_Social', 'DOOH', 'Influencer', 'Paid_Search',
              'Linear_TV', 'Partnership']
out = df[keep + media_cols]
out.to_csv('data/processed/panel_clean.csv', index=False)
out.to_pickle('data/processed/panel_clean.pkl')
json.dump(log, open('reports/tables/cleaning_log.json', 'w'), indent=1, default=str)
print(json.dumps(log, indent=1, default=str))
