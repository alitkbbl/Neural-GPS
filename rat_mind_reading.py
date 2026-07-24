"""
Rat Mind Reading: Decoding Hippocampal Place Cells
====================================================

A self-contained computational neuroscience pipeline that:
  1. Fits Poisson GLM tuning curves to simulated place-cell spike trains
  2. Quantifies position information carried by each neuron via mutual information
  3. Visualizes population activity as a low-dimensional neural manifold (PCA)
  4. Decodes the rat's position from population spiking via Bayesian filtering

Run:
    python rat_mind_reading.py

Output:
    rat_mind_reading_results.png  
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════
# GLOBAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════
N = 20                      # number of place cells
T = 1000                    # number of time steps
dt = 0.01                   # seconds per time step (=> total duration 1 s)
SIGMA = 10.0                # place field width (cm), shared by all neurons
TRACK_MIN, TRACK_MAX = 0.0, 100.0   # linear track extent (cm)

np.random.seed(42)

t_idx = np.arange(T)

# ---- Rat trajectory --------------------------------------------------
# x(t) = 50 + 50*sin(2*pi*3*t/T) + noise,  noise ~ N(0, 1)
# => ~3 full back-and-forth traversals of the 0-100 cm track
x_true = 50 + 50 * np.sin(2 * np.pi * 3 * t_idx / T) + np.random.normal(0, 1, T)
x_true = np.clip(x_true, TRACK_MIN, TRACK_MAX)  # keep the rat on the track

# ---- Place fields ------------------------------------------------------
mu = np.linspace(TRACK_MIN, TRACK_MAX, N)   # field centers, evenly spaced
c = np.log(20 * dt)                          # peak expected count/bin => 20 Hz max rate


def log_rate(x, mu_i):
    """log lambda_i(x) = c - (x - mu_i)^2 / (2*sigma^2)."""
    return c - (x - mu_i) ** 2 / (2 * SIGMA ** 2)


# lambda_i(t) for every neuron/time step -> shape (T, N)
Lambda = np.exp(log_rate(x_true[:, None], mu[None, :]))

# Spike matrix S[t, i] ~ Poisson(lambda_i(t)) -> shape (T, N)
S = np.random.poisson(Lambda)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — GLM / Tuning Curves
# ══════════════════════════════════════════════════════════════════════
n_bins_tuning = 10
bin_edges = np.linspace(TRACK_MIN, TRACK_MAX, n_bins_tuning + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_idx = np.clip(np.digitize(x_true, bin_edges) - 1, 0, n_bins_tuning - 1)

# Empirical tuning curve: mean spike count per position bin, per neuron
empirical_tuning = np.zeros((N, n_bins_tuning))
for b in range(n_bins_tuning):
    mask = bin_idx == b
    if mask.any():
        empirical_tuning[:, b] = S[mask, :].mean(axis=0)

# Poisson GLM (log link): log(rate) = b0 + b1*x + b2*x^2
# Quadratic-in-x features exactly match the algebraic form of a Gaussian
# tuning curve's log-rate, so this GLM recovers the tuning curve analytically.
X_design_raw = np.column_stack([x_true, x_true ** 2])
X_bin_design_raw = np.column_stack([bin_centers, bin_centers ** 2])

# Standardize features (x, x^2 span very different scales: ~1e2 vs ~1e4,
# which otherwise destabilizes the GLM's iterative solver).
scaler = StandardScaler()
X_design = scaler.fit_transform(X_design_raw)
X_bin_design = scaler.transform(X_bin_design_raw)

glm_tuning = np.zeros((N, n_bins_tuning))
for i in range(N):
    model = PoissonRegressor(alpha=1e-6, max_iter=1000)
    model.fit(X_design, S[:, i])
    glm_tuning[i, :] = model.predict(X_bin_design)

# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — Information Theory (Mutual Information)
# ══════════════════════════════════════════════════════════════════════
n_bins_mi = 10
pos_bin_edges = np.linspace(TRACK_MIN, TRACK_MAX, n_bins_mi + 1)
pos_bin_idx = np.clip(np.digitize(x_true, pos_bin_edges) - 1, 0, n_bins_mi - 1)

# Spike-count categories: 0, 1, 2+  (=> 3 categories)
spike_cat = np.minimum(S, 2)


def mutual_information(pos_bins, spike_bins, n_pos_bins, n_spike_bins):
    """MI(x; S_i) = sum_{x,s} p(x,s) * log2( p(x,s) / (p(x) p(s)) )."""
    T_total = len(pos_bins)
    joint = np.zeros((n_pos_bins, n_spike_bins))
    for p, s in zip(pos_bins, spike_bins):
        joint[p, s] += 1
    joint /= T_total

    p_x = joint.sum(axis=1, keepdims=True)
    p_s = joint.sum(axis=0, keepdims=True)

    valid = joint > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(joint / (p_x * p_s))
    return np.sum(joint[valid] * log_ratio[valid])


mi_values = np.array([
    mutual_information(pos_bin_idx, spike_cat[:, i], n_bins_mi, 3)
    for i in range(N)
])
mi_order = np.argsort(mi_values)[::-1]   # descending

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — Neural Manifolds (PCA)
# ══════════════════════════════════════════════════════════════════════
pca = PCA(n_components=2)
pcs = pca.fit_transform(S.astype(float))   # (T, N) -> (T, 2)

# ══════════════════════════════════════════════════════════════════════
# PHASE 4 — Bayesian Decoding
# ══════════════════════════════════════════════════════════════════════
n_grid = 100
x_grid = np.linspace(TRACK_MIN, TRACK_MAX, n_grid)
sigma_trans = 2.0   # cm, random-walk transition std

# Log-likelihood table: log_lambda_i(x_g) and lambda_i(x_g) -> shape (n_grid, N)
log_lambda_grid = log_rate(x_grid[:, None], mu[None, :])
lambda_grid = np.exp(log_lambda_grid)
lambda_sum_grid = lambda_grid.sum(axis=1)   # sum_i lambda_i(x_g), shape (n_grid,)

# Transition matrix Tmat[g, g'] = N(x_grid[g]; x_grid[g'], sigma_trans^2),
# columns normalized to sum to 1 so each source posterior maps to a valid prior.
diff = x_grid[:, None] - x_grid[None, :]
Tmat = norm.pdf(diff, loc=0, scale=sigma_trans)
Tmat /= Tmat.sum(axis=0, keepdims=True)

decoded_x = np.zeros(T)
posterior_prev = np.full(n_grid, 1.0 / n_grid)   # uniform prior at t = 0

for t in range(T):
    # Poisson log-likelihood at every grid point.
    # Note: the -log(S[t,i]!) term is constant across grid points g for a
    # fixed t, so it cancels out under softmax normalization and is omitted.
    log_lik = S[t, :] @ log_lambda_grid.T - lambda_sum_grid   # (n_grid,)

    if t == 0:
        prior = posterior_prev
    else:
        prior = Tmat @ posterior_prev
        prior /= prior.sum()

    log_post = log_lik + np.log(prior + 1e-300)
    log_post -= log_post.max()          # softmax, numerically stable
    post = np.exp(log_post)
    post /= post.sum()

    decoded_x[t] = np.sum(x_grid * post)   # posterior mean
    posterior_prev = post

rmse = np.sqrt(np.mean((decoded_x - x_true) ** 2))

# ══════════════════════════════════════════════════════════════════════
# FINAL FIGURE — 2x2 summary
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# [0,0] Phase 1: tuning curves
ax = axes[0, 0]
neurons_to_plot = [N // 5, N // 2, (4 * N) // 5]
palette = plt.cm.tab10(np.linspace(0, 1, 10))
for k, i in enumerate(neurons_to_plot):
    ax.plot(bin_centers, empirical_tuning[i], "o", color=palette[k],
            markersize=6, label=f"Neuron {i} (empirical)")
    ax.plot(bin_centers, glm_tuning[i], "-", color=palette[k],
            linewidth=2, label=f"Neuron {i} (GLM fit)")
ax.set_xlabel("Position (cm)")
ax.set_ylabel("Mean spike count / bin")
ax.set_title("Phase 1: Tuning Curves (Empirical vs. GLM)")
ax.legend(fontsize=8)

# [0,1] Phase 2: mutual information
ax = axes[0, 1]
ax.bar(range(N), mi_values[mi_order], color="teal")
ax.set_xticks(range(N))
ax.set_xticklabels(mi_order, rotation=90, fontsize=7)
ax.set_xlabel("Neuron index (sorted by MI)")
ax.set_ylabel("Mutual information (bits)")
ax.set_title("Phase 2: Mutual Information (Position vs. Spikes)")

# [1,0] Phase 3: PCA manifold
ax = axes[1, 0]
sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=x_true, cmap="viridis", s=15)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Phase 3: Neural Manifold (PCA)")
plt.colorbar(sc, ax=ax, label="True position (cm)")

# [1,1] Phase 4: Bayesian decoding
ax = axes[1, 1]
ax.plot(t_idx * dt, x_true, color="black", linewidth=1.5, label="True position")
ax.plot(t_idx * dt, decoded_x, color="crimson", linewidth=1, alpha=0.85,
        label="Decoded position")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Position (cm)")
ax.set_title(f"Phase 4: Bayesian Decoding (RMSE = {rmse:.2f} cm)")
ax.legend(fontsize=8)

fig.suptitle("Rat Mind Reading: Decoding Hippocampal Place Cells",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("rat_mind_reading_results.png", dpi=150)

print(f"Phase 4 Bayesian decoding RMSE: {rmse:.3f} cm")
print(f"Top 3 neurons by MI (descending): {mi_order[:3].tolist()}")
print(f"MI values (top 3): {np.round(mi_values[mi_order[:3]], 4).tolist()}")
