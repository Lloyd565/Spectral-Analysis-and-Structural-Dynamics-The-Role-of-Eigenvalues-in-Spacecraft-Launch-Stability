import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from scipy.integrate import odeint

def create_launch_vehicle_matrices(n_dof=5):
    """
    Create simplified mass, stiffness, and damping matrices
    for a launch vehicle structure
    
    Parameters:
    n_dof: number of degrees of freedom
    
    Returns:
    M, C, K: mass, damping, and stiffness matrices
    """
    # Mass matrix (diagonal - lumped mass model)
    # Units: kg
    M = np.diag([50000, 45000, 40000, 35000, 30000][:n_dof])
    
    # Stiffness matrix (tridiagonal - representing elastic coupling)
    # Units: N/m
    k = 1e8  # base stiffness
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        K[i, i] = 2 * k * (1 - 0.1 * i)
        if i > 0:
            K[i, i-1] = -k * (1 - 0.1 * i)
        if i < n_dof - 1:
            K[i, i+1] = -k * (1 - 0.1 * i)
    
    # Damping matrix (Rayleigh damping: C = alpha*M + beta*K)
    alpha = 0.5
    beta = 0.0001
    C = alpha * M + beta * K
    
    return M, C, K

def solve_eigenvalue_problem(M, K):
    """
    Solve generalized eigenvalue problem: Kv = λMv
    
    Returns:
    eigenvalues: natural frequencies squared (ω²)
    eigenvectors: mode shapes
    frequencies: natural frequencies in Hz
    """
    # Solve generalized eigenvalue problem
    eigenvalues, eigenvectors = eig(K, M)
    
    # Sort by eigenvalues
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate natural frequencies
    omega = np.sqrt(np.real(eigenvalues))  # rad/s
    frequencies = omega / (2 * np.pi)  # Hz
    
    return eigenvalues, eigenvectors, frequencies

def calculate_damping_ratios(M, C, K):
    """
    Calculate damping ratios from state-space eigenvalues
    """
    n = M.shape[0]
    
    # Create state-space matrix A
    M_inv = np.linalg.inv(M)
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-M_inv @ K, -M_inv @ C]
    ])
    
    # Compute eigenvalues of A
    eigenvalues_ss = np.linalg.eigvals(A)
    
    # Extract damping ratios from complex eigenvalues
    damping_ratios = []
    frequencies = []
    
    for lam in eigenvalues_ss:
        if np.imag(lam) > 0:  # Take only positive imaginary parts
            sigma = np.real(lam)
            omega = np.imag(lam)
            zeta = -sigma / np.abs(lam)
            damping_ratios.append(zeta)
            frequencies.append(omega / (2 * np.pi))
    
    return np.array(damping_ratios), np.array(frequencies)

def check_stability(M, C, K):
    """
    Check Hurwitz stability criterion: Re(λ) < 0 for all eigenvalues
    """
    n = M.shape[0]
    M_inv = np.linalg.inv(M)
    
    # State-space matrix
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-M_inv @ K, -M_inv @ C]
    ])
    
    eigenvalues = np.linalg.eigvals(A)
    real_parts = np.real(eigenvalues)
    
    is_stable = np.all(real_parts < 0)
    
    return is_stable, eigenvalues, real_parts


def pogo_frequency_separation(structural_freq, propulsion_freq):
    """
    Check NASA's frequency separation criterion for Pogo mitigation
    
    Criterion: |ω_structural - ω_propulsion| > Δω_min
    where Δω_min = 0.2 * min(ω_structural, ω_propulsion)
    """
    min_freq = min(structural_freq, propulsion_freq)
    delta_omega_min = 0.2 * min_freq
    
    separation = abs(structural_freq - propulsion_freq)
    is_safe = separation > delta_omega_min
    
    return is_safe, separation, delta_omega_min

def simulate_pogo_response(structural_freq, propulsion_freq, t_span, coupling_strength=0.1):
    """
    Simulate coupled structural-propulsion oscillation
    
    Parameters:
    structural_freq: structural natural frequency (Hz)
    propulsion_freq: propulsion system frequency (Hz)
    coupling_strength: coupling coefficient (0-1)
    """
    omega_s = 2 * np.pi * structural_freq
    omega_p = 2 * np.pi * propulsion_freq
    
    def coupled_system(y, t):
        x_s, v_s, x_p, v_p = y
        
        # Structural oscillator with propulsion coupling
        a_s = -omega_s**2 * x_s - 0.05 * omega_s * v_s + coupling_strength * omega_s**2 * x_p
        
        # Propulsion oscillator with structural coupling
        a_p = -omega_p**2 * x_p - 0.03 * omega_p * v_p + coupling_strength * omega_p**2 * x_s
        
        return [v_s, a_s, v_p, a_p]
    
    # Initial conditions: small perturbation in structural mode
    y0 = [0.01, 0, 0, 0]
    
    sol = odeint(coupled_system, y0, t_span)
    
    return sol


def construct_hankel_matrix(impulse_response, p, q):
    """
    Construct Hankel matrix from impulse response data
    
    Parameters:
    impulse_response: system output data
    p, q: dimensions of Hankel matrix
    """
    n = len(impulse_response)
    H = np.zeros((p, q))
    
    for i in range(p):
        for j in range(q):
            if i + j < n:
                H[i, j] = impulse_response[i + j]
    
    return H

def era_modal_identification(impulse_response, n_modes=3):
    """
    Eigensystem Realization Algorithm for modal parameter extraction
    """
    # Construct Hankel matrix
    n_data = len(impulse_response)
    p = n_data // 2
    q = n_data - p
    
    H = construct_hankel_matrix(impulse_response, p, q)
    
    # Singular Value Decomposition
    U, S, Vt = svd(H)
    
    # Truncate to n_modes
    U_r = U[:, :n_modes]
    S_r = np.diag(S[:n_modes])
    V_r = Vt[:n_modes, :].T
    
    # Extract system matrix (simplified)
    sqrt_S = np.sqrt(S_r)
    A_identified = np.linalg.inv(sqrt_S) @ U_r.T @ H[:, 1:] @ V_r @ np.linalg.inv(sqrt_S)
    
    # Extract modal parameters
    eigenvalues = np.linalg.eigvals(A_identified)
    
    return eigenvalues, S


def visualize_mode_shapes(eigenvectors, n_modes=3):
    """
    Visualize first n mode shapes
    """
    n_dof = eigenvectors.shape[0]
    x_positions = np.arange(n_dof)
    
    fig, axes = plt.subplots(1, n_modes, figsize=(15, 4))
    
    for i in range(n_modes):
        mode = np.real(eigenvectors[:, i])
        # Normalize
        mode = mode / np.max(np.abs(mode))
        
        axes[i].plot(x_positions, mode, 'b-o', linewidth=2, markersize=8)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].set_title(f'Mode {i+1}')
        axes[i].set_xlabel('DOF Position')
        axes[i].set_ylabel('Normalized Amplitude')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_pogo_analysis(structural_freqs, propulsion_freq=16):
    """
    Visualize frequency separation for Pogo mitigation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Frequency comparison
    modes = np.arange(1, len(structural_freqs) + 1)
    ax1.bar(modes - 0.2, structural_freqs, 0.4, label='Structural Modes', color='blue', alpha=0.7)
    ax1.axhline(y=propulsion_freq, color='red', linestyle='--', linewidth=2, 
                label=f'Propulsion Freq ({propulsion_freq} Hz)')
    
    # Add 20% safety margins
    ax1.axhline(y=propulsion_freq * 1.2, color='orange', linestyle=':', alpha=0.5, 
                label='Safety Margin (+20%)')
    ax1.axhline(y=propulsion_freq * 0.8, color='orange', linestyle=':', alpha=0.5,
                label='Safety Margin (-20%)')
    
    ax1.set_xlabel('Mode Number')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Structural vs Propulsion Frequencies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Safety assessment
    separations = []
    safety_status = []
    
    for freq in structural_freqs:
        is_safe, sep, min_sep = pogo_frequency_separation(freq, propulsion_freq)
        separations.append(sep)
        safety_status.append('Safe' if is_safe else 'Risk')
    
    colors = ['green' if s == 'Safe' else 'red' for s in safety_status]
    ax2.bar(modes, separations, color=colors, alpha=0.7)
    ax2.axhline(y=propulsion_freq * 0.2, color='black', linestyle='--', 
                label='Minimum Required Separation')
    ax2.set_xlabel('Mode Number')
    ax2.set_ylabel('Frequency Separation (Hz)')
    ax2.set_title('Pogo Safety Assessment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_stability(eigenvalues_ss):
    """
    Visualize eigenvalues in complex plane for stability check
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    real_parts = np.real(eigenvalues_ss)
    imag_parts = np.imag(eigenvalues_ss)
    
    ax.scatter(real_parts, imag_parts, c='blue', s=100, alpha=0.6, edgecolors='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stability Boundary')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Real Part', fontsize=12)
    ax.set_ylabel('Imaginary Part', fontsize=12)
    ax.set_title('Eigenvalue Distribution (Stability Analysis)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add stability region shading
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, alpha=0.2, color='green', label='Stable Region')
    ax.set_xlim(xlim)
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("LAUNCH VEHICLE STABILITY ANALYSIS")
    print("Spectral Analysis and Structural Dynamics Implementation")
    print("="*70)
    
    # Create system matrices
    M, C, K = create_launch_vehicle_matrices(n_dof=5)
    
    print("\n[1] EIGENVALUE ANALYSIS")
    print("-" * 70)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors, frequencies = solve_eigenvalue_problem(M, K)
    
    print(f"\nNatural Frequencies:")
    for i, freq in enumerate(frequencies):
        print(f"  Mode {i+1}: {freq:.4f} Hz")
    
    # Calculate damping ratios
    damping_ratios, freq_damped = calculate_damping_ratios(M, C, K)
    
    print(f"\nDamping Ratios:")
    for i, (zeta, freq) in enumerate(zip(damping_ratios, freq_damped)):
        status = "OK" if zeta > 0.03 else "Low"
        print(f"  Mode {i+1}: ζ = {zeta:.6f} (f = {freq:.4f} Hz) {status}")
    
    # Stability check
    is_stable, eigenvalues_ss, real_parts = check_stability(M, C, K)
    
    print(f"\n[2] STABILITY ANALYSIS (Hurwitz Criterion)")
    print("-" * 70)
    print(f"System Status: {'STABLE' if is_stable else 'UNSTABLE'}")
    print(f"All Re(λ) < 0: {is_stable}")
    print(f"Max Re(λ): {np.max(real_parts):.6e}")
    
    # Pogo analysis
    print(f"\n[3] POGO OSCILLATION MITIGATION")
    print("-" * 70)
    
    propulsion_freq = 16.0  # Hz (Apollo 13 case study)
    print(f"Propulsion System Frequency: {propulsion_freq} Hz")
    print(f"\nFrequency Separation Analysis:")
    
    for i, freq in enumerate(frequencies):
        is_safe, sep, min_sep = pogo_frequency_separation(freq, propulsion_freq)
        status = "SAFE" if is_safe else "RISK"
        print(f"  Mode {i+1}: Δf = {sep:.2f} Hz (min: {min_sep:.2f} Hz) - {status}")
    
    # Pogo simulation
    print(f"\n[4] POGO COUPLING SIMULATION")
    print("-" * 70)
    
    t_span = np.linspace(0, 5, 1000)
    
    # Case 1: Resonance (frequencies close)
    sol_resonance = simulate_pogo_response(16.0, 16.5, t_span, coupling_strength=0.2)
    
    # Case 2: Safe separation
    sol_safe = simulate_pogo_response(16.0, 20.0, t_span, coupling_strength=0.2)
    
    print("Simulation completed for:")
    print("  - Resonance case: f_s = 16.0 Hz, f_p = 16.5 Hz")
    print("  - Safe case: f_s = 16.0 Hz, f_p = 20.0 Hz")
    
    # Visualizations
    print(f"\n[5] GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    # Mode shapes
    fig1 = visualize_mode_shapes(eigenvectors, n_modes=3)
    fig1.savefig('mode_shapes.png', dpi=300, bbox_inches='tight')
    
    # Pogo analysis
    fig2 = visualize_pogo_analysis(frequencies, propulsion_freq)
    fig2.savefig('pogo_analysis.png', dpi=300, bbox_inches='tight')
    
    # Stability plot
    fig3 = visualize_stability(eigenvalues_ss)
    fig3.savefig('stability_analysis.png', dpi=300, bbox_inches='tight')
    
    # Pogo response comparison
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(t_span, sol_resonance[:, 0], 'r-', linewidth=2, label='Structural')
    ax1.plot(t_span, sol_resonance[:, 2], 'b--', linewidth=2, label='Propulsion')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Case 1: Resonance (16.0 Hz vs 16.5 Hz) - UNSTABLE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_span, sol_safe[:, 0], 'r-', linewidth=2, label='Structural')
    ax2.plot(t_span, sol_safe[:, 2], 'b--', linewidth=2, label='Propulsion')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement')
    ax2.set_title('Case 2: Proper Separation (16.0 Hz vs 20.0 Hz) - STABLE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig4.savefig('pogo_response.png', dpi=300, bbox_inches='tight')
    print("Pogo response comparison saved")
    
    print("\n" + "="*70)
    print("done.")
    print("="*70)
    
    plt.show()

if __name__ == "__main__":
    main()