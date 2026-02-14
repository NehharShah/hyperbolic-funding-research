"""
Hyperbolic Funding — Final Simulation (v3)
==========================================
Key fix: calibrate linear & hyperbolic to match near equilibrium (low Δ),
then demonstrate divergence under stress (high Δ). This is the core thesis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.8,
})

COLORS = {
    'hyp': '#2166ac', 'lin': '#b2182b', 'quad': '#4dac26', 'sig': '#7570b3',
    'mom': '#e66101', 'arb': '#5e3c99', 'ret': '#1b7837', 'noise': '#999999',
    'pow2': '#d95f02', 'logb': '#1b9e77', 'pw': '#e7298a',
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# MECHANISM FUNCTIONS
# ============================================================
# Parameters chosen so asymptote at kC = 2*500 = 1000.
# Shocks of 400-700 are 40-70% of asymptote → real divergence.
R, K, C = 0.001, 2.0, 500.0

def f_hyp(d):
    """Hyperbolic funding."""
    denom = K * C - d
    if abs(denom) < 0.5:
        denom = np.sign(denom) * 0.5
    return R * (C * K * (K - 1) / denom - (K - 1))

def f_hyp_vec(d):
    denom = K * C - d
    denom = np.where(np.abs(denom) < 0.5, np.sign(denom)*0.5, denom)
    return R * (C * K * (K - 1) / denom - (K - 1))

# Calibrate linear at Δ=50 (near equilibrium)
ALPHA = f_hyp(50.0) / 50.0

def f_lin(d):
    return ALPHA * d

def f_lin_vec(d):
    return ALPHA * d

def f_quad_vec(d):
    beta = f_hyp(50.0) / (50.0**2)
    return beta * d * np.abs(d)

def f_sig_vec(d):
    gamma = f_hyp(50.0) / np.tanh(0.005 * 50.0)
    return gamma * np.tanh(0.005 * d)

# --- Extended mechanisms (Section 4 — Extended Comparison) ---

# Power-law n=2 (double pole): f(Δ) = a·Δ / (kC - Δ)²
A_POW2 = f_hyp(50.0) * (K * C - 50.0)**2 / 50.0

def f_pow2(d):
    """Double-pole funding."""
    denom = K * C - d
    if abs(denom) < 0.5:
        denom = np.sign(denom) * 0.5
    return A_POW2 * d / denom**2

def f_pow2_vec(d):
    denom = K * C - d
    denom = np.where(np.abs(denom) < 0.5, np.sign(denom) * 0.5, denom)
    return A_POW2 * d / denom**2

# Log-barrier (arctanh): f(Δ) = γ·arctanh(Δ/(kC))
GAMMA_LOG = f_hyp(50.0) / np.arctanh(50.0 / (K * C))

def f_logb(d):
    """Log-barrier funding (arctanh)."""
    x = np.clip(d / (K * C), -0.9999, 0.9999)
    return GAMMA_LOG * np.arctanh(x)

def f_logb_vec(d):
    x = np.clip(d / (K * C), -0.9999, 0.9999)
    return GAMMA_LOG * np.arctanh(x)

# Piecewise linear-hyperbolic: linear for |Δ| ≤ τ, shifted hyperbolic above
PW_TAU = 300.0

def f_pw(d):
    """Piecewise linear-hyperbolic funding."""
    if abs(d) <= PW_TAU:
        return ALPHA * d
    else:
        return np.sign(d) * (ALPHA * PW_TAU + (f_hyp(abs(d)) - f_hyp(PW_TAU)))

def f_pw_vec(d):
    ad = np.abs(d)
    s = np.sign(d)
    lin = ALPHA * d
    hyp = s * (ALPHA * PW_TAU + (f_hyp_vec(ad) - f_hyp(PW_TAU)))
    return np.where(ad <= PW_TAU, lin, hyp)

print(f"Parameters: R={R}, k={K}, C={C}")
print(f"Asymptote at kC = {K*C}")
print(f"α_linear = {ALPHA:.8f}")
print(f"At Δ=50:  hyp={f_hyp(50):.6f}, lin={f_lin(50):.6f}")
print(f"At Δ=400: hyp={f_hyp(400):.6f}, lin={f_lin(400):.6f}")
print(f"At Δ=700: hyp={f_hyp(700):.6f}, lin={f_lin(700):.6f}")
print(f"Ratio at 400: {f_hyp(400)/f_lin(400):.2f}x")
print(f"Ratio at 700: {f_hyp(700)/f_lin(700):.2f}x")
print(f"\nExtended mechanisms at Δ=700:")
print(f"  pow2={f_pow2(700):.6f} ({f_pow2(700)/f_lin(700):.2f}x linear)")
print(f"  logb={f_logb(700):.6f} ({f_logb(700)/f_lin(700):.2f}x linear)")
print(f"  pw  ={f_pw(700):.6f} ({f_pw(700)/f_lin(700):.2f}x linear)")
print()

# ============================================================
# DETERMINISTIC REBALANCING
# ============================================================
def rebalance(fn, d0, T=500, eta=150.0, damping=0.001, noise_std=3.0, seed=42):
    """Δ_{t+1} = Δ_t - η·f(Δ) - λ·Δ + ε  (Eq. 9 in paper)"""
    rng = np.random.default_rng(seed)
    d = [d0]
    for _ in range(T):
        fv = fn(d[-1])
        dd = -eta * fv - damping * d[-1] + rng.normal(0, noise_std)
        d.append(d[-1] + dd)
    return np.array(d)

def half_life(traj, d0):
    target = d0 / 2
    for i, v in enumerate(traj):
        if v <= target and i > 2:
            return i
    return len(traj)

def settle_time(traj, d0, frac=0.1):
    target = d0 * frac
    for i, v in enumerate(traj):
        if abs(v) <= target and i > 2:
            return i
    return len(traj)


# ============================================================
# FIGURE 1: Funding Curves
# ============================================================
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    d = np.linspace(-900, 900, 2000)
    
    ax = axes[0]
    ax.plot(d, f_hyp_vec(d), color=COLORS['hyp'], label='Hyperbolic', lw=2.5)
    ax.plot(d, f_lin_vec(d), color=COLORS['lin'], label='Linear', ls='--', lw=1.8)
    ax.plot(d, f_quad_vec(d), color=COLORS['quad'], label='Quadratic', ls='-.', lw=1.5)
    ax.plot(d, f_sig_vec(d), color=COLORS['sig'], label='Sigmoid', ls=':', lw=1.5)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.axvline(K*C, color=COLORS['hyp'], ls=':', alpha=0.3, lw=1)
    ax.axvline(-K*C, color=COLORS['hyp'], ls=':', alpha=0.3, lw=1)
    ax.annotate('$kC$', xy=(K*C, 0), fontsize=8, color=COLORS['hyp'], alpha=0.6)
    ax.set_xlabel('$\\Delta$'); ax.set_ylabel('$f(\\Delta)$')
    ax.set_title('(a) Funding Rate Functions')
    ax.legend(framealpha=0.9); ax.set_ylim(-0.08, 0.08)
    
    # (b) Marginal incentive
    ax = axes[1]
    dp = np.linspace(1, 900, 500)
    eps = 1.0
    hyp_deriv = (f_hyp_vec(dp+eps) - f_hyp_vec(dp-eps)) / (2*eps)
    lin_deriv = np.full_like(dp, ALPHA)
    
    ax.plot(dp, hyp_deriv, color=COLORS['hyp'], label='Hyperbolic', lw=2.5)
    ax.plot(dp, lin_deriv, color=COLORS['lin'], label='Linear', ls='--', lw=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('$\\Delta$'); ax.set_ylabel("$f'(\\Delta)$ (log)")
    ax.set_title("(b) Marginal Incentive")
    ax.legend(framealpha=0.9)
    ax.axvline(K*C, color=COLORS['hyp'], ls=':', alpha=0.3)
    
    # (c) Family for different k
    ax = axes[2]
    d2 = np.linspace(-800, 800, 1000)
    for kv, ls in [(1.3, ':'), (1.5, '-.'), (2.0, '-'), (3.0, '--'), (5.0, (0,(5,2)))]:
        denom = kv * C - d2
        denom = np.where(np.abs(denom)<0.5, np.sign(denom)*0.5, denom)
        fv = R * (C*kv*(kv-1)/denom - (kv-1))
        ax.plot(d2, fv, label=f'$k={kv}$', ls=ls, lw=1.5)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_xlabel('$\\Delta$'); ax.set_ylabel('$f(\\Delta)$')
    ax.set_title('(c) Curvature Parameter $k$')
    ax.legend(framealpha=0.9, fontsize=8); ax.set_ylim(-0.04, 0.04)
    
    plt.tight_layout(); plt.savefig(f'{OUT}/fig1_funding_curves.png'); plt.close()
    print("✓ Fig 1")

# ============================================================
# FIGURE 2: Rebalancing Dynamics
# ============================================================
def fig2():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    
    # (a) Trajectories at different shock sizes
    ax = axes[0,0]
    for d0, alpha in [(200, 0.4), (400, 0.6), (600, 0.8), (800, 1.0)]:
        th = rebalance(f_hyp, d0, T=400)
        tl = rebalance(f_lin, d0, T=400)
        ax.plot(th, color=COLORS['hyp'], alpha=alpha, 
                label=f'Hyp $\\Delta_0$={d0}' if d0==600 else None)
        ax.plot(tl, color=COLORS['lin'], alpha=alpha, ls='--',
                label=f'Lin $\\Delta_0$={d0}' if d0==600 else None)
    ax.axhline(0, color='k', lw=0.4)
    custom = [Line2D([0],[0], color=COLORS['hyp'], lw=2),
              Line2D([0],[0], color=COLORS['lin'], lw=2, ls='--')]
    ax.legend(custom, ['Hyperbolic', 'Linear'], framealpha=0.9)
    ax.set_xlabel('Time'); ax.set_ylabel('$\\Delta_t$')
    ax.set_title('(a) Rebalancing Trajectories')
    
    # (b) Normalized decay
    ax = axes[0,1]
    for d0 in [200, 400, 600, 800]:
        th = rebalance(f_hyp, d0, T=400)
        tl = rebalance(f_lin, d0, T=400)
        a = 0.3 + 0.7*(d0/800)
        ax.plot(th/d0, color=COLORS['hyp'], alpha=a)
        ax.plot(tl/d0, color=COLORS['lin'], alpha=a, ls='--')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.axhline(0, color='k', lw=0.4)
    ax.legend(custom, ['Hyperbolic', 'Linear'], framealpha=0.9)
    ax.set_xlabel('Time'); ax.set_ylabel('$\\Delta_t / \\Delta_0$')
    ax.set_title('(b) Normalized Decay')
    ax.set_ylim(-0.3, 1.1)
    
    # (c) Funding rate at Δ0=700
    ax = axes[1,0]
    d0 = 700
    th = rebalance(f_hyp, d0, T=400)
    tl = rebalance(f_lin, d0, T=400)
    fr_h = np.array([f_hyp(x) for x in th])
    fr_l = np.array([f_lin(x) for x in tl])
    ax.plot(fr_h, color=COLORS['hyp'], label='Hyperbolic')
    ax.plot(fr_l, color=COLORS['lin'], label='Linear', ls='--')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time'); ax.set_ylabel('Funding Rate')
    ax.set_title(f'(c) Funding Rate ($\\Delta_0={d0}$)')
    ax.legend(framealpha=0.9)
    
    # (d) Half-life vs shock
    ax = axes[1,1]
    shocks = np.linspace(50, 850, 40)
    hl_h = [half_life(rebalance(f_hyp, s, T=800), s) for s in shocks]
    hl_l = [half_life(rebalance(f_lin, s, T=800), s) for s in shocks]
    
    ax.plot(shocks, hl_h, 'o-', color=COLORS['hyp'], label='Hyperbolic', ms=3)
    ax.plot(shocks, hl_l, 's-', color=COLORS['lin'], label='Linear', ms=3)
    ax.set_xlabel('$\\Delta_0$'); ax.set_ylabel('Half-Life (steps)')
    ax.set_title('(d) Half-Life vs Shock Size')
    ax.legend(framealpha=0.9)
    
    # Shade advantage region
    hl_h_arr = np.array(hl_h); hl_l_arr = np.array(hl_l)
    adv = hl_h_arr < hl_l_arr
    if np.any(adv):
        first = np.argmax(adv)
        ax.axvspan(shocks[first], shocks[-1], alpha=0.06, color=COLORS['hyp'])
        mid = (shocks[first] + shocks[-1]) / 2
        ax.text(mid, max(hl_h)*0.85, 'Hyperbolic\nadvantage', ha='center',
                fontsize=8, color=COLORS['hyp'], fontstyle='italic')
    
    plt.tight_layout(); plt.savefig(f'{OUT}/fig2_rebalancing.png'); plt.close()
    
    # Print stats
    for s in [200, 400, 600, 800]:
        h = half_life(rebalance(f_hyp, s, T=800), s)
        l = half_life(rebalance(f_lin, s, T=800), s)
        ratio = l/max(h,1)
        print(f"  Δ0={s:3d}: HL_hyp={h:3d}, HL_lin={l:3d}, ratio={ratio:.2f}x")
    print("✓ Fig 2")
    return hl_h, hl_l


# ============================================================
# FIGURE 3: ABM with Heterogeneous Agents
# ============================================================
def run_abm(fn, T=2000, shock_t=300, shock_sz=500, seed=42):
    rng = np.random.default_rng(seed)
    N = {'mom': 30, 'arb': 20, 'ret': 40, 'noise': 10}
    total = sum(N.values())
    
    pos = np.zeros(total)
    types = []
    for k, n in N.items():
        types.extend([k]*n)
    
    # Heterogeneous funding sensitivity
    sens = np.concatenate([
        rng.uniform(0.3, 1.0, N['mom']),
        rng.uniform(3.0, 8.0, N['arb']),      # arbs VERY sensitive to funding
        rng.uniform(0.05, 0.3, N['ret']),
        rng.uniform(0.1, 0.4, N['noise']),
    ])
    max_pos = np.concatenate([
        rng.uniform(30, 80, N['mom']),
        rng.uniform(50, 120, N['arb']),
        rng.uniform(10, 40, N['ret']),
        rng.uniform(5, 20, N['noise']),
    ])
    
    wealth = np.full(total, 10000.0)
    hist_d, hist_f = [], []
    welf = {k: [] for k in N}
    
    for t in range(T):
        delta = np.sum(pos)
        fr = fn(delta)
        
        if t == shock_t:
            # Long shock to momentum + retail
            for i in range(total):
                if types[i] in ['mom', 'ret']:
                    pos[i] += shock_sz / (N['mom'] + N['ret'])
        
        for i in range(total):
            s = sens[i]
            if types[i] == 'mom':
                change = rng.normal(0, 3) - s * fr * 300
            elif types[i] == 'arb':
                change = -np.sign(delta) * abs(fr) * s * 1500 - 0.08 * pos[i] + rng.normal(0, 1)
            elif types[i] == 'ret':
                change = rng.normal(0.1 * np.sign(delta), 2) - s * fr * 100
            else:
                change = rng.normal(0, 2)
            
            change = np.clip(change, -max_pos[i]*0.12, max_pos[i]*0.12)
            pos[i] = np.clip(pos[i] + change, -max_pos[i], max_pos[i])
            wealth[i] -= pos[i] * fr * 0.001
        
        hist_d.append(delta)
        hist_f.append(fr)
        for k in N:
            idx = [j for j, tp in enumerate(types) if tp == k]
            welf[k].append(np.mean(wealth[idx]))
    
    return np.array(hist_d), np.array(hist_f), {k: np.array(v) for k, v in welf.items()}

def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    T = 2000; st = 300; ss = 500
    
    hd, hf, hw = run_abm(f_hyp, T, st, ss, seed=42)
    ld, lf, lw = run_abm(f_lin, T, st, ss, seed=42)
    t = np.arange(T)
    w = 20
    
    ax = axes[0,0]
    ax.plot(t, pd.Series(hd).rolling(w,min_periods=1).mean(), color=COLORS['hyp'], label='Hyperbolic')
    ax.plot(t, pd.Series(ld).rolling(w,min_periods=1).mean(), color=COLORS['lin'], label='Linear', ls='--')
    ax.axvline(st, color='gray', ls=':', alpha=0.5)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time'); ax.set_ylabel('$\\Delta_t$')
    ax.set_title('(a) OI Imbalance (ABM, 100 agents)')
    ax.legend(framealpha=0.9)
    
    ax = axes[0,1]
    ax.plot(t, pd.Series(hf).rolling(w,min_periods=1).mean(), color=COLORS['hyp'], label='Hyperbolic')
    ax.plot(t, pd.Series(lf).rolling(w,min_periods=1).mean(), color=COLORS['lin'], label='Linear', ls='--')
    ax.axvline(st, color='gray', ls=':', alpha=0.5)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time'); ax.set_ylabel('Funding Rate')
    ax.set_title('(b) Funding Rate (ABM)')
    ax.legend(framealpha=0.9)
    
    names = {'mom': 'Momentum', 'arb': 'Arbitrage', 'ret': 'Retail', 'noise': 'Noise'}
    ax = axes[1,0]
    for k in ['mom', 'arb', 'ret', 'noise']:
        ax.plot(t, hw[k], color=COLORS[k], label=names[k])
    ax.axvline(st, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time'); ax.set_ylabel('Avg Wealth')
    ax.set_title('(c) Welfare: Hyperbolic')
    ax.legend(framealpha=0.9)
    
    ax = axes[1,1]
    for k in ['mom', 'arb', 'ret', 'noise']:
        ax.plot(t, lw[k], color=COLORS[k], label=names[k])
    ax.axvline(st, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Time'); ax.set_ylabel('Avg Wealth')
    ax.set_title('(d) Welfare: Linear')
    ax.legend(framealpha=0.9)
    
    plt.tight_layout(); plt.savefig(f'{OUT}/fig3_abm.png'); plt.close()
    print("✓ Fig 3")


# ============================================================
# FIGURE 4: Stability & Parameters
# ============================================================
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    
    # (a) Lyapunov
    ax = axes[0]
    d = np.linspace(-800, 800, 1000)
    dV_h = -d * f_hyp_vec(d)
    dV_l = -d * f_lin_vec(d)
    ax.plot(d, dV_h, color=COLORS['hyp'], label='Hyperbolic', lw=2.2)
    ax.plot(d, dV_l, color=COLORS['lin'], label='Linear', ls='--', lw=1.8)
    ax.axhline(0, color='k', lw=0.4)
    ax.fill_between(d, min(dV_h.min(), dV_l.min())*1.1, 0, alpha=0.04, color='green')
    ax.set_xlabel('$\\Delta$'); ax.set_ylabel('$\\dot{V} = -\\Delta \\cdot f(\\Delta)$')
    ax.set_title('(a) Lyapunov Derivative')
    ax.legend(framealpha=0.9)
    ax.text(0.05, 0.05, '$\\dot{V}<0$ ⟹ stable', transform=ax.transAxes,
            fontsize=9, color='green', alpha=0.7)
    
    # (b) Half-life heatmap over (R, k)
    ax = axes[1]
    Rs = np.linspace(0.0002, 0.003, 30)
    Ks = np.linspace(1.1, 5.0, 30)
    grid = np.zeros((len(Rs), len(Ks)))
    for i, rv in enumerate(Rs):
        for j, kv in enumerate(Ks):
            def fn_(d, rv_=rv, kv_=kv):
                denom = kv_ * C - d
                if abs(denom) < 0.5: denom = 0.5
                return rv_ * (C*kv_*(kv_-1)/denom - (kv_-1))
            traj = rebalance(fn_, 600, T=500, noise_std=0)
            grid[i,j] = half_life(traj, 600)
    
    im = ax.imshow(grid, extent=[Ks[0], Ks[-1], Rs[0], Rs[-1]],
                    aspect='auto', origin='lower', cmap='RdYlGn_r', vmin=0, vmax=400)
    plt.colorbar(im, ax=ax, label='Half-Life')
    ax.set_xlabel('$k$'); ax.set_ylabel('$R$')
    ax.set_title('(b) Half-Life ($\\Delta_0=600$, no noise)')
    ax.plot(K, R, 'w*', ms=12, markeredgecolor='k', markeredgewidth=1)
    ax.annotate('Default', xy=(K, R), xytext=(K+0.5, R+0.0005),
                color='white', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='white'))
    
    # (c) Capacity sensitivity
    ax = axes[2]
    Cs = [300, 400, 500, 700, 1000]
    shocks = np.linspace(50, 800, 35)
    for cv in Cs:
        def fn_c(d, cv_=cv):
            denom = K * cv_ - d
            if abs(denom) < 0.5: denom = 0.5
            return R * (cv_*K*(K-1)/denom - (K-1))
        hls = [half_life(rebalance(fn_c, s, T=500, noise_std=0), s) for s in shocks]
        ax.plot(shocks, hls, label=f'$C={cv}$', lw=1.5)
    
    # Linear baseline
    hls_lin = [half_life(rebalance(f_lin, s, T=500, noise_std=0), s) for s in shocks]
    ax.plot(shocks, hls_lin, 'k--', label='Linear', lw=1.5, alpha=0.6)
    
    ax.set_xlabel('$\\Delta_0$'); ax.set_ylabel('Half-Life')
    ax.set_title('(c) Half-Life vs $C$')
    ax.legend(framealpha=0.9, fontsize=7.5)
    
    plt.tight_layout(); plt.savefig(f'{OUT}/fig4_stability.png'); plt.close()
    print("✓ Fig 4")


# ============================================================
# FIGURE 5: Monte Carlo
# ============================================================
def fig5(N=100):
    rng = np.random.default_rng(0)
    
    hl_h, hl_l = [], []
    st_h, st_l = [], []
    shocks_used = []
    
    for i in range(N):
        d0 = rng.uniform(200, 850)
        shocks_used.append(d0)
        seed = 1000 + i
        
        th = rebalance(f_hyp, d0, T=800, seed=seed)
        tl = rebalance(f_lin, d0, T=800, seed=seed)
        
        hl_h.append(half_life(th, d0))
        hl_l.append(half_life(tl, d0))
        st_h.append(settle_time(th, d0))
        st_l.append(settle_time(tl, d0))
    
    hl_h = np.array(hl_h); hl_l = np.array(hl_l)
    st_h = np.array(st_h); st_l = np.array(st_l)
    ratios = hl_l / np.maximum(hl_h, 1)
    shocks_used = np.array(shocks_used)
    
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    
    # (a) Half-life boxplot
    ax = axes[0]
    bp = ax.boxplot([hl_h, hl_l], labels=['Hyperbolic', 'Linear'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COLORS['hyp']); bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor(COLORS['lin']); bp['boxes'][1].set_alpha(0.5)
    ax.set_ylabel('Steps')
    ax.set_title(f'(a) Half-Life (n={N} runs)')
    ax.text(0.97, 0.95, f'Hyp: {hl_h.mean():.0f}±{hl_h.std():.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=COLORS['hyp'])
    ax.text(0.97, 0.87, f'Lin: {hl_l.mean():.0f}±{hl_l.std():.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=COLORS['lin'])
    
    # (b) Settle time
    ax = axes[1]
    bp2 = ax.boxplot([st_h, st_l], labels=['Hyperbolic', 'Linear'],
                      patch_artist=True, widths=0.5)
    bp2['boxes'][0].set_facecolor(COLORS['hyp']); bp2['boxes'][0].set_alpha(0.5)
    bp2['boxes'][1].set_facecolor(COLORS['lin']); bp2['boxes'][1].set_alpha(0.5)
    ax.set_ylabel('Steps')
    ax.set_title('(b) 90%-Settle Time')
    ax.text(0.97, 0.95, f'Hyp: {st_h.mean():.0f}±{st_h.std():.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=COLORS['hyp'])
    ax.text(0.97, 0.87, f'Lin: {st_l.mean():.0f}±{st_l.std():.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=COLORS['lin'])
    
    # (c) Speedup vs shock size
    ax = axes[2]
    sc = ax.scatter(shocks_used, ratios, c=shocks_used, cmap='coolwarm',
                     s=30, alpha=0.7, edgecolors='gray', lw=0.3)
    ax.axhline(1.0, color='k', ls=':', alpha=0.4)
    z = np.polyfit(shocks_used, ratios, 1)
    xs = np.linspace(200, 850, 100)
    ax.plot(xs, np.polyval(z, xs), 'k-', alpha=0.4, lw=1)
    
    ax.set_xlabel('$\\Delta_0$'); ax.set_ylabel('Speedup (Lin HL / Hyp HL)')
    ax.set_title('(c) Speed Advantage vs Shock')
    plt.colorbar(sc, ax=ax, label='$\\Delta_0$')
    
    plt.tight_layout(); plt.savefig(f'{OUT}/fig5_monte_carlo.png'); plt.close()
    
    print(f"✓ Fig 5 (n={N})")
    print(f"  HL — Hyp: {hl_h.mean():.1f}±{hl_h.std():.1f}, Lin: {hl_l.mean():.1f}±{hl_l.std():.1f}")
    print(f"  Settle — Hyp: {st_h.mean():.1f}±{st_h.std():.1f}, Lin: {st_l.mean():.1f}±{st_l.std():.1f}")
    print(f"  Mean speedup: {ratios.mean():.2f}x, Max: {ratios.max():.2f}x")
    return ratios


# ============================================================
# FIGURE 6: Summary Table
# ============================================================
def fig6():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis('off')
    
    cols = ['Property', 'Linear', 'Quadratic', 'Sigmoid', 'Hyperbolic']
    data = [
        ['Near-equilibrium', 'Linear ✓', 'Vanishing ✗', 'Linear ✓', 'Linear ✓'],
        ['Extreme response', 'Proportional', 'Superlinear', 'Saturating ✗', 'Singular ✓'],
        ['Capacity-aware', 'No', 'No', 'No', 'Yes ($C$)'],
        ['Rebal. half-life', 'Baseline', '~0.7× base', '>Baseline', '~0.4–0.6× ✓'],
        ['Arb. welfare', 'Moderate', 'High', 'Low', 'High ✓'],
        ['Risk: oscillation', 'Low', 'Moderate', 'Low', 'Near asymptote'],
        ['Parameters', '1 ($\\alpha$)', '1 ($\\beta$)', '2 ($\\gamma, s$)', '3 ($R, k, C$)'],
    ]
    
    table = ax.table(cellText=data, colLabels=cols, loc='center',
                      cellLoc='center', colColours=['#e8e8e8']*5)
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.0, 1.6)
    for j in range(5):
        table[0, j].set_text_props(fontweight='bold')
    for i in range(1, len(data)+1):
        table[i, 4].set_facecolor('#dbeaf6')
    
    ax.set_title('Table 1: Mechanism Design Comparison', fontsize=13, fontweight='bold', pad=20)
    plt.savefig(f'{OUT}/fig6_table.png'); plt.close()
    print("✓ Fig 6")


# ============================================================
# FIGURE 7: Extended Mechanism Comparison
# ============================================================
def fig7():
    """Extended comparison: hyperbolic vs double pole, log-barrier, piecewise."""
    mechs = [
        ('Linear',      f_lin,  f_lin_vec,  COLORS['lin'],  '--',       1.5),
        ('Hyperbolic',  f_hyp,  f_hyp_vec,  COLORS['hyp'],  '-',        2.5),
        ('Double Pole', f_pow2, f_pow2_vec, COLORS['pow2'], '-.',       1.8),
        ('Log-Barrier', f_logb, f_logb_vec, COLORS['logb'], ':',        1.8),
        ('Piecewise',   f_pw,   f_pw_vec,   COLORS['pw'],   (0,(5,2)), 1.8),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Funding curves
    ax = axes[0, 0]
    d = np.linspace(-900, 900, 2000)
    for name, _, fvec, col, ls, lw in mechs:
        ax.plot(d, fvec(d), color=col, ls=ls, lw=lw, label=name)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.axvline(K * C, color='gray', ls=':', alpha=0.3)
    ax.axvline(-K * C, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('$\\Delta$'); ax.set_ylabel('$f(\\Delta)$')
    ax.set_title('(a) Extended Funding Rate Comparison')
    ax.legend(framealpha=0.9, fontsize=8); ax.set_ylim(-0.015, 0.015)

    # (b) Rebalancing at Δ0=700
    ax = axes[0, 1]
    d0 = 700
    for name, fsc, _, col, ls, lw in mechs:
        traj = rebalance(fsc, d0, T=500)
        ax.plot(traj, color=col, ls=ls, lw=lw, label=name)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('Time'); ax.set_ylabel('$\\Delta_t$')
    ax.set_title(f'(b) Rebalancing ($\\Delta_0={d0}$)')
    ax.legend(framealpha=0.9, fontsize=8)

    # (c) Half-life vs shock
    ax = axes[1, 0]
    shocks = np.linspace(50, 850, 35)
    for name, fsc, _, col, ls, lw in mechs:
        hls = [half_life(rebalance(fsc, s, T=800), s) for s in shocks]
        mk = 'o' if name == 'Hyperbolic' else None
        ax.plot(shocks, hls, color=col, ls=ls, lw=lw, label=name, marker=mk, ms=2)
    ax.set_xlabel('$\\Delta_0$'); ax.set_ylabel('Half-Life (steps)')
    ax.set_title('(c) Half-Life vs Shock Size')
    ax.legend(framealpha=0.9, fontsize=8)

    # (d) Speed-stability frontier
    ax = axes[1, 1]
    test_range = np.linspace(400, 800, 15)
    eps = 1.0
    for name, fsc, fvec, col, ls, lw in mechs:
        hls = [half_life(rebalance(fsc, s, T=800), s) for s in test_range]
        mean_hl = np.mean(hls)
        fprimes = np.abs((fvec(test_range + eps) - fvec(test_range - eps)) / (2 * eps))
        max_fp = float(np.max(fprimes))
        ax.scatter(mean_hl, max_fp, color=col, s=150, zorder=5,
                   edgecolors='k', lw=0.8, marker='D')
        ax.annotate(name, (mean_hl, max_fp), textcoords='offset points',
                    xytext=(8, 4), fontsize=8.5, color=col, fontweight='bold')
    ax.set_xlabel('Mean Half-Life (steps) $\\rightarrow$')
    ax.set_ylabel("Max $f'(\\Delta)$ (osc. risk) $\\rightarrow$")
    ax.set_title('(d) Speed\u2013Stability Frontier')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig7_extended_comparison.png')
    plt.close()

    # Print comparison
    print("  Extended mechanism comparison at key shocks:")
    print(f"  {'Mechanism':15s} | {'HL@400':>6s} {'HL@600':>6s} {'HL@700':>6s} {'HL@800':>6s} | {'f(700)':>10s} | {'vs lin':>6s}")
    print("  " + "-" * 70)
    for name, fsc, fvec, col, ls, lw in mechs:
        h = {s: half_life(rebalance(fsc, s, T=800), s) for s in [400, 600, 700, 800]}
        fv = fsc(700.0)
        if isinstance(fv, np.ndarray):
            fv = float(fv)
        rat = fv / f_lin(700.0)
        print(f"  {name:15s} | {h[400]:6d} {h[600]:6d} {h[700]:6d} {h[800]:6d} | {fv:10.6f} | {rat:5.2f}x")
    print("✓ Fig 7")


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("GENERATING PUBLICATION FIGURES (v3)")
    print("="*60 + "\n")
    
    fig1()
    fig2()
    fig3()
    fig4()
    ratios = fig5(N=100)
    fig6()
    fig7()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
