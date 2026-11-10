"""Check the JAX two-state Kalman filter (local level + time-varying price coefficient) against
statsmodels' state-space implementation on simulated data with known parameters."""
import sys
import numpy as np
import jax.numpy as jnp
import statsmodels.api as sm

sys.path.insert(0, 'src')
import hbnn  # noqa: E402

rng = np.random.default_rng(0)
T, rho, rl, s_eta, s_l, sig = 120, 0.9, 0.97, 0.4, 0.15, 0.1
q, ql = (1 - rho ** 2) * s_eta ** 2, (1 - rl ** 2) * s_l ** 2
x = rng.normal(0, 0.3, T)
eta, lev = np.zeros(T), np.zeros(T)
eta[0], lev[0] = rng.normal(0, s_eta), rng.normal(0, s_l)
for t in range(1, T):
    eta[t] = rho * eta[t - 1] + rng.normal(0, np.sqrt(q))
    lev[t] = rl * lev[t - 1] + rng.normal(0, np.sqrt(ql))
y = lev + x * eta + rng.normal(0, sig, T)
X = np.zeros((1, T, hbnn.P), np.float32); X[0, :, hbnn.I_BP] = x
f32 = jnp.float32
out = hbnn.kalman(f32(rho), f32(rl), jnp.array([ql], f32), jnp.full((1, T), q, f32), jnp.asarray(X),
                  jnp.asarray(y[None], f32), jnp.zeros((1, T)), jnp.array([sig], f32), jnp.ones((1, T), bool),
                  jnp.array([s_eta], f32))
mine = float(out[-1].sum())
mod = sm.tsa.statespace.MLEModel(y, k_states=2)
Zd = np.zeros((1, 2, T)); Zd[0, 0] = 1; Zd[0, 1] = x
mod['design'] = Zd
mod['transition'] = np.diag([rl, rho]); mod['selection'] = np.eye(2)
mod['state_cov'] = np.diag([ql, q]); mod['obs_cov'] = np.array([[sig ** 2]])
mod.initialize_known(np.zeros(2), np.diag([s_l ** 2, s_eta ** 2]))
ref = float(mod.ssm.loglike())
print(f'JAX filter loglik {mine:.4f}  statsmodels {ref:.4f}  diff {abs(mine - ref):.2e}')
assert abs(mine - ref) < 1e-2
