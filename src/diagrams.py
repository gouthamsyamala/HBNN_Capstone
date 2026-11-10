"""Architecture and workflow diagrams for the report (reports/figures/fig_workflow.png, fig_architecture.png)."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.use('Agg')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})
DARK, MID, LIGHT, GREY = '#1f4e79', '#4f81bd', '#dce6f2', '#f2f2f2'


def box(ax, x, y, w, h, title, body='', fc=LIGHT, ec=DARK, tc='k', fs=8, bs=6.6, gap=0.085):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.015', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 0.035, title, ha='center', va='top', fontsize=fs, weight='bold', color=tc)
    if body:
        ax.text(x + w / 2, y + h - gap, body, ha='center', va='top', fontsize=bs, color=tc, linespacing=1.3)


def arrow(ax, a, b, style='-|>', color=DARK, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=1,
                                 connectionstyle=f'arc3,rad={rad}'))


# --------------------------------------------------------------------------- workflow
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
stages = [('1. Ingestion', 'licensed data cube\n(7,498 rows × 75 cols);\nprivate masking map', 'prep.py'),
          ('2. Cleaning', 'structural checks, base-\nprice restatement, missing\nvalues, exclusion flags', 'prep.py'),
          ('3. Features', 'log prices, price ratio,\nmechanic support, TPD,\ncompetitor index, adstock', 'prep.py'),
          ('4. EDA', 'promotion share, tiers,\nprice paths, sparsity,\nmedia, seasonality', 'eda.py'),
          ('5. Models', 'M0 OLS · M1 HB (Gibbs)\nM2 HBNN · M3 HBNN-KF\n(SVI in NumPyro / JAX)', 'models.py, hbnn.py'),
          ('6. Evaluation', 'rolling-origin backtest,\n4 origins × 13 weeks; WMAPE,\nMASE, CRPS, ELPD; RQ tests', 'backtest*.py, evaluate_final.py'),
          ('7. Outputs', 'elasticity paths, lift by\nmechanic, media ROI, volume\ndecomposition, scenarios', 'final_analysis.py')]
w, h = 0.215, 0.25
pos = [(0.01 + i * 0.25, 0.64) for i in range(4)] + [(0.135 + i * 0.25, 0.27) for i in range(3)]
for i, ((t, b_, scr), (x, y)) in enumerate(zip(stages, pos)):
    box(ax, x, y, w, h, t, b_, fc='#c6d9f1' if i in (4, 5) else LIGHT, fs=7.2, bs=5.6, gap=0.075)
    ax.text(x + w / 2, y - 0.025, scr, ha='center', va='top', fontsize=6.5, style='italic', color='#404040')
    if i in (0, 1, 2, 4, 5):
        arrow(ax, (x + w, y + h / 2), (pos[i + 1][0], y + h / 2))
arrow(ax, (0.76 + w / 2, 0.64 - 0.05), (0.135 + w / 2, 0.27 + h + 0.005), rad=-0.05)
box(ax, 0.01, 0.01, 0.55, 0.14, 'configs/config.yaml + Makefile',
    'thresholds, backtest origins, priors, SVI steps, seed 640;\n`make all` reruns every stage on the licensed data', fc=GREY, ec='#7f7f7f', fs=7.2, bs=6)
box(ax, 0.60, 0.01, 0.39, 0.14, 'Public synthetic panel',
    'synthetic.py → recovery.py (`make demo`):\nknown elasticities, no licensed data needed', fc=GREY, ec='#7f7f7f', fs=7.2, bs=6)
ax.text(0.5, 0.995, 'Licensed data stay local (data/raw and data/processed are git-ignored); code, configuration, '
        'aggregate tables and figures are public.', ha='center', va='top', fontsize=6.8, color='#404040')
fig.savefig('reports/figures/fig_workflow.png', dpi=220, bbox_inches='tight')
plt.close()

# --------------------------------------------------------------------------- HBNN-KF architecture
fig, ax = plt.subplots(figsize=(7.5, 4.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
box(ax, 0.01, 0.70, 0.20, 0.24, 'PPG attributes z', 'brand, pack-size band,\nprice tier, retailer\n(one-hot, 17 inputs)')
box(ax, 0.27, 0.70, 0.21, 0.24, 'Hierarchy encoder', 'Bayesian MLP 17→8→6\n(tanh); offsets for price,\nprice-ratio and mechanic\ncoefficients')
box(ax, 0.54, 0.70, 0.22, 0.24, 'Coefficient prior', 'category μ₀ → brand μ_b\n+ encoder offset → series b_j\n(partial pooling, centred)')
box(ax, 0.80, 0.70, 0.195, 0.24, 'Constraints', 'β, γ = −softplus(·)\nmedia λ = softplus(·)\n(own brands only)')
arrow(ax, (0.21, 0.82), (0.27, 0.82)); arrow(ax, (0.48, 0.82), (0.54, 0.82)); arrow(ax, (0.76, 0.82), (0.80, 0.82))
box(ax, 0.01, 0.36, 0.20, 0.25, 'Weekly drivers x_t', 'ln base price, ln price\nratio, mechanic support,\nln TPD, ln comp. price,\nadstock media, season')
box(ax, 0.27, 0.36, 0.21, 0.25, 'Static mean μ_jt', 'x_t′ b_j\n(Equation 1)', fc='#c6d9f1')
box(ax, 0.01, 0.04, 0.20, 0.25, 'Drivers d_t', '|Δ ln TPD|,\ncompetitor promo depth,\nconsumer sentiment')
box(ax, 0.27, 0.04, 0.21, 0.25, 'Process-noise net', 'MLP 3→4→1, tanh-\nbounded; scales the\nstate variance by\ne⁻¹ to e¹')
box(ax, 0.54, 0.16, 0.25, 0.45, 'Kalman layer (differentiable)',
    'state = (level ℓ_t, elasticity η_t)\nℓ_t = ρ_ℓ ℓ_t−1 + u_t\nη_t = ρ η_t−1 + w_t\n'
    'y_t = μ_jt + ℓ_t + x_bp,t η_t + ε_t\n\njax.lax.scan over 139 weeks,\n61 series in parallel;\n'
    'marginal likelihood → ELBO', fc='#c6d9f1')
box(ax, 0.83, 0.36, 0.16, 0.25, 'Outputs', 'forecasts with\nintervals; smoothed\nβ_jt = β_j + η_t;\nlift, ROI, scenarios', fc=GREY,
    ec='#7f7f7f')
box(ax, 0.83, 0.04, 0.16, 0.25, 'Inference', 'SVI (Adam, 8,000\nsteps, mean-field\nguide); NUTS check\non 10% subsample', fc=GREY,
    ec='#7f7f7f')
arrow(ax, (0.21, 0.485), (0.27, 0.485)); arrow(ax, (0.48, 0.485), (0.54, 0.485))
arrow(ax, (0.21, 0.165), (0.27, 0.165)); arrow(ax, (0.48, 0.165), (0.54, 0.25))
arrow(ax, (0.65, 0.70), (0.65, 0.61)); arrow(ax, (0.375, 0.70), (0.375, 0.61))
arrow(ax, (0.79, 0.485), (0.83, 0.485)); arrow(ax, (0.79, 0.25), (0.83, 0.165), style='<|-|>', color='#7f7f7f')
fig.savefig('reports/figures/fig_architecture.png', dpi=220, bbox_inches='tight')
plt.close()
print('diagrams written')
