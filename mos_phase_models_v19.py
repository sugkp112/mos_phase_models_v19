import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
GRID = 80
N_MOSQ = 250
T_STEPS = 600
RUNS = 50  # Number of Monte Carlo runs for statistical robustness

# REPRODUCIBILITY CONTROL
# For scientific rigor: Set to an integer for exact reproducibility
# For robustness testing: Set to None to verify conclusion stability across runs
MASTER_SEED = None  # Recommended: None for final publication to show robustness
                    # Use fixed seed (e.g., 42) only for debugging

HOST = np.array([GRID // 2, GRID // 2])
HOST_RADIUS = 2.0

CO2_BASE = 1.0
HEAT_BASE = 1.0
W_CO2 = 1.0
W_HEAT = 1.0
SENSORY_NOISE = 0.40

# Pre-computed distance field
xx, yy = np.meshgrid(np.arange(GRID), np.arange(GRID))
dist_to_host = np.sqrt((xx - HOST[0])**2 + (yy - HOST[1])**2)

# Initialize global RNG for seed generation
if MASTER_SEED is not None:
    _seed_rng = np.random.default_rng(MASTER_SEED)
    print(f"🔒 FIXED SEED MODE: Master seed = {MASTER_SEED}")
    print("   ⚠️  Results will be identical across runs (good for debugging)")
else:
    _seed_rng = np.random.default_rng()
    print("🎲 ROBUSTNESS MODE: No fixed seed (recommended)")
    print("   ✅ Results demonstrate statistical robustness")


# =========================================================
# FIELD GENERATORS
# =========================================================
def build_fields(n_fake_sources=0):
    """Build CO2 and thermal fields"""
    co2 = CO2_BASE / (1.0 + dist_to_host)
    heat = HEAT_BASE / (1.0 + dist_to_host)

    if n_fake_sources > 0:
        radius = 35
        angles = np.linspace(0, 2*np.pi, n_fake_sources, endpoint=False)
        for a in angles:
            fx = int(HOST[0] + radius*np.cos(a))
            fy = int(HOST[1] + radius*np.sin(a))
            fx = np.clip(fx, 0, GRID-1)
            fy = np.clip(fy, 0, GRID-1)
            dist_fake = np.sqrt((xx - fx)**2 + (yy - fy)**2)
            heat += 0.8 / (1.0 + dist_fake)

    return co2, heat


# =========================================================
# VELOCITY FIELD CALCULATORS
# =========================================================
def calc_wind_velocity(pos, strength):
    """Calculate wind field velocity (upward flow)"""
    n = pos.shape[0]
    return np.zeros(n), np.full(n, -strength)


def calc_vortex_velocity(pos, strength):
    """Calculate vortex field velocity (circular flow)"""
    vec = pos - HOST
    r = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
    unit = vec / r
    perp = np.stack([-unit[:,1], unit[:,0]], axis=1)
    v = strength * perp
    return v[:,0], v[:,1]


def calc_zero_velocity(pos):
    """Zero velocity field"""
    return np.zeros(pos.shape[0]), np.zeros(pos.shape[0])


# =========================================================
# CORE SIMULATION ENGINE
# =========================================================
def simulate_once_worker(params):
    """
    Picklable worker function for parallel execution
    params: (co2, heat, field_type, field_strength, n_steps, seed, noise, n_mosq)
    field_type: 'wind', 'vortex', 'zero', 'combo'
    """
    co2, heat, field_type, field_params, n_steps, seed, noise, n_mosq = params
    
    rng = np.random.default_rng(seed)
    total = W_CO2 * co2 + W_HEAT * heat

    pos = rng.integers(0, GRID, size=(n_mosq, 2))
    alive = np.ones(n_mosq, bool)
    reached = 0

    for t in range(n_steps):
        cx = pos[:,0]
        cy = pos[:,1]

        # Gradient calculation
        gx = total[(cx+1) % GRID, cy] - total[(cx-1) % GRID, cy]
        gy = total[cx, (cy+1) % GRID] - total[cx, (cy-1) % GRID]
        grad_move = np.stack([gx, gy], axis=1)

        # Velocity field calculation
        if field_type == 'wind':
            vx, vy = calc_wind_velocity(pos, field_params)
        elif field_type == 'vortex':
            vx, vy = calc_vortex_velocity(pos, field_params)
        elif field_type == 'zero':
            vx, vy = calc_zero_velocity(pos)
        elif field_type == 'combo':
            vx1, vy1 = calc_vortex_velocity(pos, field_params[0])
            vx2, vy2 = calc_wind_velocity(pos, field_params[1])
            vx, vy = vx1 + vx2, vy1 + vy2
        else:
            vx, vy = 0, 0
            
        flow_move = np.stack([vx, vy], axis=1)
        noise_move = noise * rng.normal(size=grad_move.shape)

        move = grad_move + flow_move + noise_move
        pos = pos + np.sign(move).astype(int)
        pos = np.clip(pos, 0, GRID-1)

        # Check if reached host
        d = np.linalg.norm(pos - HOST, axis=1)
        hit = (d <= HOST_RADIUS) & alive
        reached += hit.sum()
        alive[hit] = False

        if not alive.any():
            break

    return reached / n_mosq


# =========================================================
# PARALLEL EXECUTION UTILITIES
# =========================================================
def run_parallel_sims(param_list, max_workers=8):
    """Execute parallel simulations using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(simulate_once_worker, param_list))
    return results


def scan_parameter_curve(param_values, field_type, n_fake_sources=0, 
                         label="", n_steps=T_STEPS, n_runs=RUNS):
    """Scan parameter space and compute statistics"""
    co2, heat = build_fields(n_fake_sources)
    means, stds = [], []
    all_seeds = []  # Store seeds for reproducibility

    for p in param_values:
        # Generate deterministic seeds from global RNG
        seeds = [_seed_rng.integers(0, 10**9) for _ in range(n_runs)]
        all_seeds.append(seeds[:3])  # Store first 3 for logging
        
        param_list = [(co2, heat, field_type, p, n_steps, seed, SENSORY_NOISE, N_MOSQ) 
                      for seed in seeds]
        rates = run_parallel_sims(param_list, max_workers=8)
        
        m, s = float(np.mean(rates)), float(np.std(rates))
        means.append(m)
        stds.append(s)
        print(f"{label}={p:.2f}  mean={m:.3f}±{s:.3f}  seeds={seeds[:3]}...")

    return np.array(means), np.array(stds), all_seeds


# =========================================================
# VISUALIZATION: SCHEMA DIAGRAMS
# =========================================================
def add_schema_inset(ax, schema_type):
    """Add schema diagram to visualize experimental setup"""
    from matplotlib.patches import Circle, FancyArrow, Wedge
    
    # Create inset axes
    inset_ax = ax.inset_axes([0.68, 0.65, 0.28, 0.28])
    inset_ax.set_xlim(-1.2, 1.2)
    inset_ax.set_ylim(-1.2, 1.2)
    inset_ax.set_aspect('equal')
    inset_ax.axis('off')
    
    # Draw host (center)
    host_circle = Circle((0, 0), 0.15, color='red', alpha=0.8, label='Host')
    inset_ax.add_patch(host_circle)
    
    if schema_type == 'wind':
        # Draw upward wind arrows
        for x in [-0.8, -0.4, 0, 0.4, 0.8]:
            arrow = FancyArrow(x, -1.0, 0, 0.6, width=0.08, 
                              head_width=0.15, head_length=0.15,
                              fc='blue', ec='blue', alpha=0.6)
            inset_ax.add_patch(arrow)
        inset_ax.text(0, -1.15, 'Wind Flow', ha='center', fontsize=8, style='italic')
        
    elif schema_type == 'vortex':
        # Draw circular vortex
        theta = np.linspace(0, 2*np.pi, 16, endpoint=False)
        for t in theta:
            r = 0.8
            x, y = r*np.cos(t), r*np.sin(t)
            dx, dy = -np.sin(t)*0.25, np.cos(t)*0.25
            arrow = FancyArrow(x, y, dx, dy, width=0.06,
                              head_width=0.12, head_length=0.1,
                              fc='purple', ec='purple', alpha=0.6)
            inset_ax.add_patch(arrow)
        inset_ax.text(0, -1.15, 'Vortex Field', ha='center', fontsize=8, style='italic')
        
    elif schema_type == 'decoy':
        # Draw decoy sources
        decoy_angles = [0, 2*np.pi/3, 4*np.pi/3]
        for angle in decoy_angles:
            x, y = 0.7*np.cos(angle), 0.7*np.sin(angle)
            decoy = Circle((x, y), 0.1, color='orange', alpha=0.7)
            inset_ax.add_patch(decoy)
        inset_ax.text(0, -1.15, 'Thermal Decoys', ha='center', fontsize=8, style='italic')
        
    elif schema_type == 'combo':
        # Combined wind + vortex
        # Vortex (smaller)
        theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
        for t in theta:
            r = 0.6
            x, y = r*np.cos(t), r*np.sin(t)
            dx, dy = -np.sin(t)*0.15, np.cos(t)*0.15
            arrow = FancyArrow(x, y, dx, dy, width=0.04,
                              head_width=0.08, head_length=0.08,
                              fc='purple', ec='purple', alpha=0.5)
            inset_ax.add_patch(arrow)
        # Wind (background)
        for x in [-0.9, 0, 0.9]:
            arrow = FancyArrow(x, -1.0, 0, 0.5, width=0.06,
                              head_width=0.1, head_length=0.1,
                              fc='blue', ec='blue', alpha=0.4)
            inset_ax.add_patch(arrow)
        inset_ax.text(0, -1.15, 'Combined Fields', ha='center', fontsize=8, style='italic')
    
    # Add mosquito symbols
    for angle in [np.pi/6, 3*np.pi/4, 5*np.pi/3]:
        x, y = 1.0*np.cos(angle), 1.0*np.sin(angle)
        inset_ax.plot(x, y, 'k^', markersize=6, alpha=0.7)
    
    inset_ax.text(0, 1.15, 'Experimental Setup', ha='center', 
                 fontsize=9, weight='bold')


# =========================================================
# FIGURE GENERATION FUNCTIONS
# =========================================================
def generate_figure_wind(save_dir="figures"):
    """Generate Figure 1: Wind strength effect"""
    print("\n🔬 Generating Figure 1: Wind Strength Analysis...")
    
    wind_vals = np.linspace(0, 3, 16)
    means, stds, seeds = scan_parameter_curve(wind_vals, 'wind', label="wind", n_runs=RUNS)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(wind_vals, means, yerr=stds, marker='o', capsize=4, 
                linewidth=2.5, markersize=8, color='#2E86AB', 
                elinewidth=1.5, label='Success Rate')
    ax.axhline(0.01, color='#E63946', ls='--', linewidth=2, 
              alpha=0.8, label='Threshold (1%)')
    
    # Add shaded region below threshold
    ax.fill_between(wind_vals, 0, 0.01, alpha=0.1, color='red')
    
    ax.set_title('Effect of Wind Strength on Mosquito Navigation Success', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Wind Strength (arbitrary units)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-0.02, max(means)*1.15)
    
    # Add schema
    add_schema_inset(ax, 'wind')
    
    # Add annotations for key points
    collapse_idx = np.where(means < 0.01)[0]
    if len(collapse_idx) > 0:
        first_collapse = collapse_idx[0]
        ax.annotate(f'Collapse at {wind_vals[first_collapse]:.2f}',
                   xy=(wind_vals[first_collapse], means[first_collapse]),
                   xytext=(wind_vals[first_collapse]+0.5, means[first_collapse]+0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, color='red', weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Fig1_wind_strength.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/Fig1_wind_strength.png")
    plt.close()


def generate_figure_vortex(save_dir="figures"):
    """Generate Figure 2: Vortex chaos effect"""
    print("\n🔬 Generating Figure 2: Vortex Chaos Analysis...")
    
    k_vals = np.linspace(0, 3, 16)
    means, stds, seeds = scan_parameter_curve(k_vals, 'vortex', label="vortex", n_runs=RUNS)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(k_vals, means, yerr=stds, marker='s', capsize=4, 
                linewidth=2.5, markersize=8, color='#A23B72',
                elinewidth=1.5, label='Success Rate')
    ax.axhline(0.01, color='#E63946', ls='--', linewidth=2, 
              alpha=0.8, label='Threshold (1%)')
    
    ax.fill_between(k_vals, 0, 0.01, alpha=0.1, color='red')
    
    ax.set_title('Effect of Vortex Strength on Mosquito Navigation Success', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Vortex Strength (arbitrary units)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-0.02, max(means)*1.15)
    
    add_schema_inset(ax, 'vortex')
    
    collapse_idx = np.where(means < 0.01)[0]
    if len(collapse_idx) > 0:
        first_collapse = collapse_idx[0]
        ax.annotate(f'Collapse at {k_vals[first_collapse]:.2f}',
                   xy=(k_vals[first_collapse], means[first_collapse]),
                   xytext=(k_vals[first_collapse]+0.5, means[first_collapse]+0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, color='red', weight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Fig2_vortex_chaos.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/Fig2_vortex_chaos.png")
    plt.close()


def generate_figure_decoy(save_dir="figures"):
    """Generate Figure 3: Thermal decoy dilution"""
    print("\n🔬 Generating Figure 3: Thermal Decoy Analysis...")
    
    fake_vals = np.arange(0, 9)
    means, stds = [], []
    
    for n_fake in fake_vals:
        co2, heat = build_fields(n_fake)
        seeds = [_seed_rng.integers(0, 10**9) for _ in range(RUNS)]
        param_list = [(co2, heat, 'zero', None, T_STEPS, seed, SENSORY_NOISE, N_MOSQ) 
                      for seed in seeds]
        rates = run_parallel_sims(param_list, max_workers=8)
        means.append(np.mean(rates))
        stds.append(np.std(rates))
        print(f"fake_sources={n_fake}  mean={means[-1]:.3f}±{stds[-1]:.3f}  seeds={seeds[:3]}...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(fake_vals, means, yerr=stds, marker='^', capsize=4, 
                linewidth=2.5, markersize=9, color='#F77F00',
                elinewidth=1.5, label='Success Rate')
    ax.axhline(0.01, color='#E63946', ls='--', linewidth=2, 
              alpha=0.8, label='Threshold (1%)')
    
    ax.fill_between(fake_vals, 0, 0.01, alpha=0.1, color='red')
    
    ax.set_title('Effect of Thermal Decoy Sources on Mosquito Host-Seeking', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Number of Fake Heat Sources', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-0.02, max(means)*1.15)
    
    add_schema_inset(ax, 'decoy')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Fig3_thermal_decoy.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/Fig3_thermal_decoy.png")
    plt.close()


def generate_figure_combined_heatmap(save_dir="figures"):
    """Generate Figure 4: Combined aerodynamic phase map"""
    print("\n🔬 Generating Figure 4: Combined Phase Map...")
    
    wind_vals = np.linspace(0, 2.5, 10)
    vort_vals = np.linspace(0, 2.5, 10)
    co2, heat = build_fields(n_fake_sources=2)
    H = np.zeros((len(vort_vals), len(wind_vals)))
    
    for i, v in enumerate(vort_vals):
        for j, w in enumerate(wind_vals):
            seeds = [_seed_rng.integers(0, 10**9) for _ in range(RUNS)]
            param_list = [(co2, heat, 'combo', (v, w), T_STEPS, seed, SENSORY_NOISE, N_MOSQ) 
                          for seed in seeds]
            rates = run_parallel_sims(param_list, max_workers=8)
            H[i, j] = np.mean(rates)
            
            if H[i, j] < 0.01:
                print(f"🔥 Collapse @ vortex={v:.2f}, wind={w:.2f}, mean={H[i,j]:.3f}")
    
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(H, origin='lower', aspect='auto', cmap='RdYlGn',
                   extent=[wind_vals[0], wind_vals[-1], vort_vals[0], vort_vals[-1]],
                   vmin=0, vmax=np.max(H))
    
    cbar = plt.colorbar(im, ax=ax, label='Success Rate', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Add contour lines
    contours = ax.contour(wind_vals, vort_vals, H, levels=[0.01, 0.05, 0.1, 0.2], 
                         colors='black', linewidths=1.5, alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=9, fmt='%.2f')
    
    ax.set_title('Combined Aerodynamic Invisibility Phase Diagram\n(with 2 thermal decoys)', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Wind Strength', fontsize=12)
    ax.set_ylabel('Vortex Strength', fontsize=12)
    
    add_schema_inset(ax, 'combo')
    
    # Highlight collapse region
    collapse_region = H < 0.01
    if np.any(collapse_region):
        ax.contour(wind_vals, vort_vals, collapse_region, levels=[0.5], 
                  colors='red', linewidths=3, linestyles='--')
        ax.text(0.05, 0.95, 'Red dashed: Collapse region (< 1%)',
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Fig4_combined_phase_map.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/Fig4_combined_phase_map.png")
    plt.close()


def generate_supp_figure_time_convergence(save_dir="figures"):
    """Supplementary Figure S1: Time horizon convergence"""
    print("\n🧪 Generating Supplementary Figure S1...")
    
    steps_list = [300, 600, 900]
    means, stds = [], []
    
    co2, heat = build_fields()
    for steps in steps_list:
        seeds = [_seed_rng.integers(0, 10**9) for _ in range(30)]
        param_list = [(co2, heat, 'vortex', 2.0, steps, seed, SENSORY_NOISE, N_MOSQ) 
                      for seed in seeds]
        rates = run_parallel_sims(param_list, max_workers=8)
        means.append(np.mean(rates))
        stds.append(np.std(rates))
        print(f"T={steps} mean={means[-1]:.4f}±{stds[-1]:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(steps_list, means, yerr=stds, marker='o', capsize=5, 
                linewidth=2.5, markersize=9, color='#06A77D', elinewidth=1.5)
    ax.axhline(means[-1], color='gray', ls=':', linewidth=1.5, 
              alpha=0.6, label=f'Converged value: {means[-1]:.4f}')
    
    ax.set_title('Temporal Convergence Test\n(Vortex Strength = 2.0)', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Simulation Time Steps', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Add convergence annotation
    diff = abs(means[-1] - means[-2])
    ax.text(0.95, 0.05, f'Convergence: Δ={diff:.5f}',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS1_time_convergence.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/FigS1_time_convergence.png")
    plt.close()


def generate_supp_figure_runs_convergence(save_dir="figures"):
    """Supplementary Figure S2: Monte Carlo convergence"""
    print("\n🧪 Generating Supplementary Figure S2...")
    
    wind_vals = [0.3, 1.5]
    run_levels = [20, 50, 100]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    co2, heat = build_fields()
    
    colors = ['#2E86AB', '#A23B72']
    for idx, wind in enumerate(wind_vals):
        means_list = []
        for R in run_levels:
            seeds = [_seed_rng.integers(0, 10**9) for _ in range(R)]
            param_list = [(co2, heat, 'wind', wind, T_STEPS, seed, SENSORY_NOISE, N_MOSQ) 
                          for seed in seeds]
            rates = run_parallel_sims(param_list, max_workers=8)
            means_list.append(np.mean(rates))
            print(f"wind={wind:.2f}, RUNS={R} → {means_list[-1]:.4f}")
        
        ax.plot(run_levels, means_list, marker='o', label=f'Wind = {wind}', 
               linewidth=2.5, markersize=9, color=colors[idx])
    
    ax.set_title('Monte Carlo Convergence Analysis', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Number of Simulation Runs', fontsize=12)
    ax.set_ylabel('Mean Success Rate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS2_runs_convergence.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/FigS2_runs_convergence.png")
    plt.close()


def generate_supp_figure_sensitivity(save_dir="figures"):
    """Supplementary Figure S3: Parameter sensitivity"""
    print("\n🧪 Generating Supplementary Figure S3...")
    
    noise_list = [0.2, 0.4, 0.6]
    pop_list = [150, 250, 400]
    data = np.zeros((len(noise_list), len(pop_list)))
    
    co2, heat = build_fields()
    for i, noise in enumerate(noise_list):
        for j, N in enumerate(pop_list):
            seeds = [_seed_rng.integers(0, 10**9) for _ in range(20)]
            param_list = [(co2, heat, 'vortex', 2.0, T_STEPS, seed, noise, N) 
                          for seed in seeds]
            rates = run_parallel_sims(param_list, max_workers=8)
            data[i, j] = np.mean(rates)
            print(f"noise={noise} N={N} → {data[i,j]:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='viridis', origin='lower',
                   extent=[pop_list[0]-25, pop_list[-1]+25, 
                          noise_list[0]-0.05, noise_list[-1]+0.05],
                   aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, label='Mean Success Rate')
    cbar.ax.tick_params(labelsize=10)
    
    # Add value annotations
    for i in range(len(noise_list)):
        for j in range(len(pop_list)):
            text = ax.text(pop_list[j], noise_list[i], f'{data[i, j]:.3f}',
                          ha="center", va="center", color="white", 
                          fontsize=10, weight='bold')
    
    ax.set_title('Model Sensitivity to Noise and Population Size\n(Vortex Strength = 2.0)', 
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Mosquito Population Size', fontsize=12)
    ax.set_ylabel('Sensory Noise Amplitude', fontsize=12)
    ax.set_xticks(pop_list)
    ax.set_yticks(noise_list)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS3_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/FigS3_sensitivity_analysis.png")
    plt.close()


def simulate_with_custom_field(total_field, wind_strength, seed):
    """Helper for weight sensitivity with custom total field"""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, GRID, size=(N_MOSQ, 2))
    alive = np.ones(N_MOSQ, bool)
    reached = 0
    
    for t in range(T_STEPS):
        cx, cy = pos[:,0], pos[:,1]
        gx = total_field[(cx+1) % GRID, cy] - total_field[(cx-1) % GRID, cy]
        gy = total_field[cx, (cy+1) % GRID] - total_field[cx, (cy-1) % GRID]
        grad_move = np.stack([gx, gy], axis=1)
        
        vx, vy = calc_wind_velocity(pos, wind_strength)
        flow_move = np.stack([vx, vy], axis=1)
        noise_move = SENSORY_NOISE * rng.normal(size=grad_move.shape)
        
        move = grad_move + flow_move + noise_move
        pos = pos + np.sign(move).astype(int)
        pos = np.clip(pos, 0, GRID-1)
        
        d = np.linalg.norm(pos - HOST, axis=1)
        hit = (d <= HOST_RADIUS) & alive
        reached += hit.sum()
        alive[hit] = False
        
        if not alive.any():
            break
    
    return reached / N_MOSQ


def run_parallel_sims_with_radius(param_list, max_workers=8):
    """Worker for radius sensitivity (9 params including radius)"""
    def worker_with_radius(params):
        co2, heat, field_type, field_params, n_steps, seed, noise, n_mosq, radius = params
        rng = np.random.default_rng(seed)
        total = W_CO2 * co2 + W_HEAT * heat
        pos = rng.integers(0, GRID, size=(n_mosq, 2))
        alive = np.ones(n_mosq, bool)
        reached = 0
        
        for t in range(n_steps):
            cx, cy = pos[:,0], pos[:,1]
            gx = total[(cx+1) % GRID, cy] - total[(cx-1) % GRID, cy]
            gy = total[cx, (cy+1) % GRID] - total[cx, (cy-1) % GRID]
            grad_move = np.stack([gx, gy], axis=1)
            
            if field_type == 'wind':
                vx, vy = calc_wind_velocity(pos, field_params)
            else:
                vx, vy = 0, 0
            
            flow_move = np.stack([vx, vy], axis=1)
            noise_move = noise * rng.normal(size=grad_move.shape)
            move = grad_move + flow_move + noise_move
            pos = pos + np.sign(move).astype(int)
            pos = np.clip(pos, 0, GRID-1)
            
            d = np.linalg.norm(pos - HOST, axis=1)
            hit = (d <= radius) & alive  # Use custom radius
            reached += hit.sum()
            alive[hit] = False
            
            if not alive.any():
                break
        
        return reached / n_mosq
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_with_radius, param_list))
    return results


def simulate_and_track_times(co2, heat, field_type, field_strength, seed):
    """Track success times for each mosquito"""
    rng = np.random.default_rng(seed)
    total = W_CO2 * co2 + W_HEAT * heat
    pos = rng.integers(0, GRID, size=(N_MOSQ, 2))
    alive = np.ones(N_MOSQ, bool)
    success_times = []
    
    for t in range(T_STEPS):
        cx, cy = pos[:,0], pos[:,1]
        gx = total[(cx+1) % GRID, cy] - total[(cx-1) % GRID, cy]
        gy = total[cx, (cy+1) % GRID] - total[cx, (cy-1) % GRID]
        grad_move = np.stack([gx, gy], axis=1)
        
        if field_type == 'wind':
            vx, vy = calc_wind_velocity(pos, field_strength)
        elif field_type == 'vortex':
            vx, vy = calc_vortex_velocity(pos, field_strength)
        elif field_type == 'zero':
            vx, vy = calc_zero_velocity(pos)
        else:
            vx, vy = 0, 0
        
        flow_move = np.stack([vx, vy], axis=1)
        noise_move = SENSORY_NOISE * rng.normal(size=grad_move.shape)
        move = grad_move + flow_move + noise_move
        pos = pos + np.sign(move).astype(int)
        pos = np.clip(pos, 0, GRID-1)
        
        d = np.linalg.norm(pos - HOST, axis=1)
        hit = (d <= HOST_RADIUS) & alive
        
        # Record success times
        for idx in np.where(hit)[0]:
            success_times.append(t)
        
        alive[hit] = False
        if not alive.any():
            break
    
    return success_times


def generate_supp_figure_weight_sensitivity(save_dir="figures"):
    """Supplementary Figure S4: CO2/Heat weight sensitivity"""
    print("\n🧪 Generating Supplementary Figure S4...")
    
    weight_ratios = [(1.0, 0.5), (1.0, 1.0), (1.0, 1.5), (0.5, 1.0), (1.5, 1.0)]
    labels = ['CO2:Heat=2:1', 'Equal', 'CO2:Heat=1:1.5', 
              'CO2:Heat=1:2', 'CO2:Heat=1.5:1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    co2_base, heat_base = build_fields()
    
    for idx, (w_co2, w_heat) in enumerate(weight_ratios):
        means, stds = [], []
        wind_vals = np.linspace(0, 3, 12)
        
        for wind in wind_vals:
            total = w_co2 * co2_base + w_heat * heat_base
            seeds = [_seed_rng.integers(0, 10**9) for _ in range(30)]
            # Modified simulate_once_worker to accept custom weights
            rates = []
            for seed in seeds:
                # Run simulation with custom weight field
                rate = simulate_with_custom_field(total, wind, seed)
                rates.append(rate)
            means.append(np.mean(rates))
            stds.append(np.std(rates))
        
        ax.plot(wind_vals, means, marker='o', label=labels[idx], linewidth=2)
    
    ax.axhline(0.01, color='red', ls='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Wind Strength', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Sensitivity to CO2/Heat Weighting', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS4_weight_sensitivity.png", dpi=300)
    print(f"✅ Saved: {save_dir}/FigS4_weight_sensitivity.png")
    plt.close()


def generate_supp_figure_radius_sensitivity(save_dir="figures"):
    """Supplementary Figure S5: Host capture radius sensitivity"""
    print("\n🧪 Generating Supplementary Figure S5...")
    
    radius_vals = [1.0, 2.0, 3.0, 4.0]
    colors = ['#E63946', '#F77F00', '#06A77D', '#2E86AB']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    co2, heat = build_fields()
    
    for idx, radius in enumerate(radius_vals):
        wind_vals = np.linspace(0, 3, 12)
        means, stds = [], []
        
        for wind in wind_vals:
            seeds = [_seed_rng.integers(0, 10**9) for _ in range(30)]
            param_list = [(co2, heat, 'wind', wind, T_STEPS, seed, 
                          SENSORY_NOISE, N_MOSQ, radius)  # Add radius param
                          for seed in seeds]
            rates = run_parallel_sims_with_radius(param_list, max_workers=8)
            means.append(np.mean(rates))
            stds.append(np.std(rates))
        
        ax.plot(wind_vals, means, marker='s', label=f'Radius = {radius}', 
               linewidth=2.5, color=colors[idx])
    
    ax.axhline(0.01, color='red', ls='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('Wind Strength', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Sensitivity to Host Capture Radius', fontsize=14, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS5_radius_sensitivity.png", dpi=300)
    print(f"✅ Saved: {save_dir}/FigS5_radius_sensitivity.png")
    plt.close()


def generate_supp_figure_success_distribution(save_dir="figures"):
    """Supplementary Figure S6: Distribution of success times"""
    print("\n🧪 Generating Supplementary Figure S6...")
    
    conditions = [
        ('Baseline', 'zero', 0),
        ('Wind=1.0', 'wind', 1.0),
        ('Vortex=1.0', 'vortex', 1.0),
        ('Wind=2.0', 'wind', 2.0)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    co2, heat = build_fields()
    
    for idx, (label, field_type, strength) in enumerate(conditions):
        success_times = []
        
        # Run multiple simulations and collect success times
        for _ in range(20):
            seed = _seed_rng.integers(0, 10**9)
            times = simulate_and_track_times(co2, heat, field_type, 
                                            strength, seed)
            success_times.extend(times)
        
        if len(success_times) > 0:
            axes[idx].hist(success_times, bins=30, alpha=0.7, 
                          color='steelblue', edgecolor='black')
            axes[idx].axvline(np.mean(success_times), color='red', 
                             ls='--', linewidth=2, 
                             label=f'Mean={np.mean(success_times):.1f}')
            axes[idx].set_xlabel('Time to Success (steps)', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(label, fontsize=12, weight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Success Times Across Conditions', 
                fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/FigS6_time_distribution.png", dpi=300)
    print(f"✅ Saved: {save_dir}/FigS6_time_distribution.png")
    plt.close()



# =========================================================
# MASTER GENERATION FUNCTION
# =========================================================
def generate_all_figures(save_dir="figures"):
    """Generate all figures (main + supplementary)"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Main figures
    generate_figure_wind(save_dir)
    generate_figure_vortex(save_dir)
    generate_figure_decoy(save_dir)
    generate_figure_combined_heatmap(save_dir)
    
    # Supplementary figures
    generate_supp_figure_time_convergence(save_dir)
    generate_supp_figure_runs_convergence(save_dir)
    generate_supp_figure_sensitivity(save_dir)
    generate_supp_figure_weight_sensitivity(save_dir)      # ✅ ADD
    generate_supp_figure_radius_sensitivity(save_dir)      # ✅ ADD
    generate_supp_figure_success_distribution(save_dir)    # ✅ ADD



# =========================================================
# ROBUSTNESS VERIFICATION UTILITY
# =========================================================
def verify_robustness(n_repetitions=3):
    """
    Verify that conclusions are robust across multiple runs with different seeds.
    This demonstrates that results are not artifacts of specific random seeds.
    """
    print("\n" + "=" * 70)
    print("🔬 ROBUSTNESS VERIFICATION TEST")
    print("=" * 70)
    print(f"Running key experiments {n_repetitions} times with different random seeds...")
    print("If conclusions are robust, the critical thresholds should be consistent.\n")
    
    # Test 1: Wind collapse point
    print("📊 Test 1: Wind-induced collapse threshold")
    wind_collapse_points = []
    
    for run in range(n_repetitions):
        print(f"\n  Run {run+1}/{n_repetitions}:")
        global _seed_rng
        _seed_rng = np.random.default_rng()  # New seed each time
        
        wind_vals = np.linspace(0, 3, 16)
        means, stds, _ = scan_parameter_curve(wind_vals, 'wind', 
                                              label=f"  wind_run{run+1}", n_runs=30)
        
        # Find collapse point (success rate < 1%)
        collapse_idx = np.where(means < 0.01)[0]
        if len(collapse_idx) > 0:
            collapse_point = wind_vals[collapse_idx[0]]
            wind_collapse_points.append(collapse_point)
            print(f"  → Collapse at wind ≈ {collapse_point:.2f}")
        else:
            print(f"  → No collapse detected")
    
    if len(wind_collapse_points) > 0:
        mean_collapse = np.mean(wind_collapse_points)
        std_collapse = np.std(wind_collapse_points)
        print(f"\n  📈 SUMMARY: Collapse threshold = {mean_collapse:.2f} ± {std_collapse:.3f}")
        print(f"     Coefficient of Variation = {std_collapse/mean_collapse*100:.1f}%")
        
        if std_collapse/mean_collapse < 0.1:
            print(f"     ✅ ROBUST: Variation < 10%, conclusion is reliable")
        else:
            print(f"     ⚠️  Consider increasing RUNS for better stability")
    
    # Test 2: Vortex collapse point
    print("\n\n📊 Test 2: Vortex-induced collapse threshold")
    vortex_collapse_points = []
    
    for run in range(n_repetitions):
        print(f"\n  Run {run+1}/{n_repetitions}:")
        _seed_rng = np.random.default_rng()
        
        vortex_vals = np.linspace(0, 3, 16)
        means, stds, _ = scan_parameter_curve(vortex_vals, 'vortex',
                                              label=f"  vortex_run{run+1}", n_runs=30)
        
        collapse_idx = np.where(means < 0.01)[0]
        if len(collapse_idx) > 0:
            collapse_point = vortex_vals[collapse_idx[0]]
            vortex_collapse_points.append(collapse_point)
            print(f"  → Collapse at vortex ≈ {collapse_point:.2f}")
        else:
            print(f"  → No collapse detected")
    
    if len(vortex_collapse_points) > 0:
        mean_collapse = np.mean(vortex_collapse_points)
        std_collapse = np.std(vortex_collapse_points)
        print(f"\n  📈 SUMMARY: Collapse threshold = {mean_collapse:.2f} ± {std_collapse:.3f}")
        print(f"     Coefficient of Variation = {std_collapse/mean_collapse*100:.1f}%")
        
        if std_collapse/mean_collapse < 0.1:
            print(f"     ✅ ROBUST: Variation < 10%, conclusion is reliable")
        else:
            print(f"     ⚠️  Consider increasing RUNS for better stability")
    
    print("\n" + "=" * 70)
    print("🎯 CONCLUSION:")
    print("   If both tests show < 10% variation, your results are statistically robust")
    print("   and not artifacts of specific parameter choices or random seeds.")
    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 MOSQUITO PHASE MODEL - Publication Figure Generator")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Grid Size: {GRID}x{GRID}")
    print(f"   Mosquito Agents: {N_MOSQ}")
    print(f"   Simulation Steps: {T_STEPS}")
    print(f"   Monte Carlo Runs: {RUNS}")
    print(f"   Sensory Noise: {SENSORY_NOISE}")
    print(f"⚡ Parallel Processing: ThreadPoolExecutor (8 workers)")
    
    if MASTER_SEED is not None:
        print(f"🔒 REPRODUCIBILITY: Fixed seed = {MASTER_SEED}")
        print(f"   → Results will be identical across runs")
    else:
        print(f"🎲 ROBUSTNESS MODE: No fixed seed")
        print(f"   → Run multiple times to verify conclusion stability!")
    
    print("=" * 70)
    
    if "--sup" in sys.argv or "-s" in sys.argv:
        print("\n📈 Generating SUPPLEMENTARY FIGURES only...")
        Path("figures").mkdir(exist_ok=True)
        generate_supp_figure_time_convergence("figures")
        generate_supp_figure_runs_convergence("figures")
        generate_supp_figure_sensitivity("figures")
        
    elif "--main" in sys.argv or "-m" in sys.argv:
        print("\n📈 Generating MAIN FIGURES only...")
        Path("figures").mkdir(exist_ok=True)
        generate_figure_wind("figures")
        generate_figure_vortex("figures")
        generate_figure_decoy("figures")
        generate_figure_combined_heatmap("figures")
        
    elif "--wind" in sys.argv:
        print("\n📈 Generating WIND figure only...")
        Path("figures").mkdir(exist_ok=True)
        generate_figure_wind("figures")
        
    elif "--vortex" in sys.argv:
        print("\n📈 Generating VORTEX figure only...")
        Path("figures").mkdir(exist_ok=True)
        generate_figure_vortex("figures")
        
    elif "--decoy" in sys.argv:
        print("\n📈 Generating DECOY figure only...")
        Path("figures").mkdir(exist_ok=True)
        generate_figure_decoy("figures")
        
    elif "--phase" in sys.argv:
        print("\n📈 Generating PHASE MAP figure only...")
        Path("figures").mkdir(exist_ok=True)
        generate_figure_combined_heatmap("figures")
    
    elif "--verify" in sys.argv or "--robust" in sys.argv:
        print("\n🔬 Running ROBUSTNESS VERIFICATION...")
        # Check if user specified number of repetitions
        n_reps = 3
        for arg in sys.argv:
            if arg.startswith("--reps="):
                n_reps = int(arg.split("=")[1])
        verify_robustness(n_repetitions=n_reps)
        
    else:
        print("\n📈 Generating ALL FIGURES...")
        generate_all_figures("figures")
    
    print("\n" + "=" * 70)
    print("✨ Figure generation completed successfully!")
    print(f"📁 Output directory: ./figures/")
    print("=" * 70)
    print("\n💡 Usage tips:")
    print("   --main or -m     : Generate main figures only (Fig1-4)")
    print("   --sup or -s      : Generate supplementary figures only (FigS1-3)")
    print("   --wind           : Generate wind figure only")
    print("   --vortex         : Generate vortex figure only")
    print("   --decoy          : Generate decoy figure only")
    print("   --phase          : Generate phase map only")
    print("   --verify         : Run robustness verification test")
    print("   (no arguments)   : Generate all figures")
    print("=" * 70)

