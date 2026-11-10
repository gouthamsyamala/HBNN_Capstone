"""Synthetic CPG category panel with known (ground-truth) elasticities.

Fully simulated: no value is copied or resampled from the licensed data. The panel has the same
columns that backtest_hbnn.py, final_analysis.py and recovery.py read, so the models can be run end to
end without access to the licensed cube. Half of the series have a static base-price elasticity; the
other half have a time-varying elasticity (either a step change at week 80 or a mean-reverting drift).

Writes data/synthetic/panel_synthetic.csv and data/synthetic/truth.csv.
Usage: python src/synthetic.py [seed]
"""
import os
import sys

import numpy as np
import pandas as pd

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 640
rng = np.random.default_rng(seed)
T = 139
weeks = pd.date_range('2024-01-07', periods=T, freq='7D')
BRANDS = {'Own Brand A': (1.75, 1), 'Own Brand B': (2.60, 1), 'Competitor C': (1.80, 0),
          'Competitor D': (4.90, 0), 'Competitor E': (2.40, 0), 'Competitor F': (5.20, 0),
          'Private Label': (1.35, 0)}
PACKS = {'14-17': 1.25, '21.5-25': 1.0, '40-45': 0.8, '66-67': 0.7}
MECH = ['TPR', 'FO', 'DO', 'F&D']
THETA = {'TPR': 0.05, 'FO': 0.12, 'DO': 0.30, 'F&D': 0.35}       # true support effects (log scale)
brand_beta = {b: rng.uniform(-2.2, -1.2) for b in BRANDS}
os.makedirs('data/synthetic', exist_ok=True)

t = np.arange(T)
season = 0.10 * np.sin(2 * np.pi * t / 52.18 + 1.2) + 0.03 * np.cos(4 * np.pi * t / 52.18)
thanks = np.array([(w.month == 11) and (22 <= w.day <= 28) for w in weeks], float)
xmas = np.array([(w.month == 12) and (22 <= w.day <= 28) for w in weeks], float)
easter = np.isin(weeks, pd.to_datetime(['2024-03-31', '2025-04-20', '2026-04-05'])).astype(float)
csi = 60 + np.cumsum(rng.normal(0, 0.8, T))
comp = {m: np.cumsum(rng.normal(0.001, 0.01, T)) for m in ['Retailer 1', 'Retailer 2']}
media = {}
for b, (_, own) in BRANDS.items():
    if own:
        spend = rng.lognormal(np.log(4e5 if b.endswith('A') else 1.5e5), 0.45, T)
        ad = np.zeros(T)
        for k in range(T):
            ad[k] = spend[k] + (0.6 * ad[k - 1] if k else 0)
        media[b] = (spend, np.log1p(ad / 1e5))

rows, truth = [], []
sid = 0
for market in ['Retailer 1', 'Retailer 2']:
    for b, (px, own) in BRANDS.items():
        for pack, pf in rng.choice(list(PACKS.items()), size=3 if b in ('Own Brand A', 'Competitor C', 'Private Label') else 2,
                                   replace=False):
            pf = float(pf)
            sid += 1
            s = f'S{"1" if market.endswith("1") else "2"}-{sid:02d}'
            start = int(rng.choice([0] * 9 + [100]))                     # ~10% launches -> naturally sparse
            n = T - start
            # base price: list-price increases at 1-3 random weeks plus small noise
            lbp = np.log(px * pf) + np.zeros(T)
            for k in rng.choice(np.arange(10, T - 10), size=rng.integers(1, 4), replace=False):
                lbp[k:] += rng.uniform(0.03, 0.08)
            lbp += rng.normal(0, 0.005, T)
            # promotions
            promo = rng.random(T) < 0.32
            mech = np.where(promo, rng.choice(MECH, T, p=[.65, .17, .09, .09]), 'None')
            depth = np.where(promo, rng.uniform(0.10, 0.35, T), 0.0)
            supp = np.where(promo, rng.uniform(0.6, 1.0, T), 0.0)
            S = {m: np.where(mech == m, supp, 0.0) for m in MECH}
            lnPR = np.log(1 - depth * supp)
            lntpd = np.log(rng.uniform(150, 900)) + np.cumsum(rng.normal(0, 0.01, T))
            if start:
                lntpd[start:start + 8] -= np.linspace(1.5, 0, 8)
            # true coefficients
            beta0 = brand_beta[b] + rng.normal(0, 0.2)
            kind = rng.choice(['static', 'step', 'drift'], p=[.5, .25, .25])
            eta = np.zeros(T)
            if kind == 'step':
                eta[80:] = rng.choice([-1, 1], p=[.75, .25]) * rng.uniform(0.4, 0.7)
            elif kind == 'drift':
                for k in range(1, T):
                    eta[k] = 0.97 * eta[k - 1] + rng.normal(0, 0.06)
            gamma = -rng.uniform(1.2, 2.0)
            theta = {m: THETA[m] + rng.normal(0, 0.04) for m in MECH}
            delta, phi = rng.uniform(1.0, 1.4), rng.uniform(0.05, 0.3)
            lam = 0.03 if own else 0.0
            lnA = media[b][1] if own else np.zeros(T)
            lnCP = np.log(2.5) + comp[market]
            mu = (rng.uniform(8, 11) + (beta0 + eta) * (lbp - lbp.mean()) + gamma * lnPR
                  + sum(theta[m] * S[m] for m in MECH) + delta * (lntpd - lntpd.mean()) + phi * (lnCP - lnCP.mean())
                  + lam * (lnA - lnA.mean()) + season - 0.17 * thanks + 0.05 * xmas + 0.03 * easter
                  + 0.0005 * t)
            lnV = mu + rng.normal(0, 0.08, T)
            V = np.exp(lnV)
            price = np.exp(lbp) * (1 - depth * supp)
            for k in range(start, T):
                rows.append({'series_id': s, 'market': market, 'brand': b, 'own': own,
                             'sku': f'{b[-1]} ITEM {pack}OZ', 'PackSize': pack, 'week': weeks[k], 't': k,
                             'Volume_Sales': V[k], 'Dollar_Sales': V[k] * price[k], 'lnV': lnV[k],
                             'Price_per_Volume': price[k], 'PriceBase_Volume': np.exp(lbp[k]),
                             'promo_ratio': 1 - depth[k] * supp[k], 'lnBP': lbp[k], 'lnPR': lnPR[k],
                             's_tpr': S['TPR'][k], 's_fo': S['FO'][k], 's_do': S['DO'][k], 's_fd': S['F&D'][k],
                             'lnTPD': lntpd[k], 'Total_Points_of_Distribution': np.exp(lntpd[k]), 'acv': 90.0,
                             'comp_price': np.exp(lnCP[k]), 'lnCP': lnCP[k],
                             'Total_BrandMedia_Spends': media[b][0][k] if own else 0.0, 'lnA': lnA[k],
                             'csi': csi[k], 'post_break': 0,
                             'sin1': np.sin(2 * np.pi * k / 52.18), 'cos1': np.cos(2 * np.pi * k / 52.18),
                             'sin2': np.sin(4 * np.pi * k / 52.18), 'cos2': np.cos(4 * np.pi * k / 52.18),
                             'hol_thanks': thanks[k], 'hol_xmas': xmas[k], 'hol_easter': easter[k],
                             'promo_event': mech[k], 'sparse_flag': int(n < 52), 'archived': 0,
                             'low_acv_week': 0, 'outlier': 0})
                truth.append({'series_id': s, 't': k, 'kind': kind, 'beta_t': beta0 + eta[k], 'beta_static': beta0,
                              'gamma': gamma, **{f'theta_{m}': theta[m] for m in MECH}, 'lambda': lam})
pd.DataFrame(rows).to_csv('data/synthetic/panel_synthetic.csv', index=False)
pd.DataFrame(truth).to_csv('data/synthetic/truth.csv', index=False)
print(f'{sid} series, {len(rows)} rows written to data/synthetic/')
