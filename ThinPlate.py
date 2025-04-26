# -*- coding: utf-8 -*-
"""
Core analysis logic for thin plate bending.
@author: Murat GÜVEN
"""
import pandas as pd
import numpy as np
from numpy import sin, cos, pi, unravel_index
from scipy.integrate import solve_bvp
from scipy.special import binom

class Plate:
    """Represents the thin plate and its properties."""
    def __init__(self, a, b, e, t0=0.05, E=200e9, nu=0.3, Q=-10e3):
        # Geometric properties
        self.a = a              # Plate length in x-direction (m)
        self.b = b              # Plate length in y-direction (m)
        self.t0 = t0            # Plate thickness @ y = b/2 (m)
        self.e = e              # Small parameter for thickness variation [0, 1)

        # Material properties
        self.E = E              # Modulus of elasticity (N/m^2)
        self.nu = nu            # Poisson's ratio

        # Flexural rigidity @ y = b/2 (Nm)
        self.D0 = self.E * self.t0**3 / (12 * (1 - self.nu**2))

        # Transverse load intensity (N/m^2)
        self.Q = Q              # minus sign indicates the direction

    def thickness(self, y):
        """Calculates thickness at a given y-coordinate."""
        return self.t0 * (1 + self.e * (2 * y / self.b - 1))

    def flexural_rigidity(self, y):
        """Calculates flexural rigidity at a given y-coordinate."""
        return self.E * self.thickness(y)**3 / (12 * (1 - self.nu**2))


def Di(i, y, plate): 
    return plate.D0 * binom(3, i) * ((2 * y / plate.b - 1))**i

def Diy(i, y, plate, n_nodesY):
    if i >= 1:
        return plate.D0 * binom(3, i) * i * ((2 * y / plate.b - 1))**(i - 1) * (2 / plate.b)
    else:
        return np.zeros((n_nodesY,))

def Diyy(i, y, plate, n_nodesY):
    if i >= 2:
        return plate.D0 * binom(3, i) * i * (i - 1) * ((2 * y / plate.b - 1))**(i - 2) * (2 / plate.b)**2
    else:
        return np.zeros((n_nodesY,))


def Analyze(plate, n_nodesX, n_nodesY, N_max=100, k_max=5, m_tol=1e-6, k_tol=1e-5):
    """
    Analyzes the plate deflection and related quantities using perturbation
    and Fourier series with convergence checks. Also returns convergence history.

    Args:
        plate (Plate): Plate object containing geometric and material properties.
        n_nodesX (int): Number of nodes in x-direction.
        n_nodesY (int): Number of nodes in y-direction.
        N_max (int): Maximum number of Fourier terms.
        k_max (int): Maximum perturbation order.
        m_tol (float): Tolerance for Fourier series convergence.
        k_tol (float): Tolerance for perturbation series convergence.

    Returns:
        tuple: (results, m_convergence_history, k_convergence_history)
               results: Array containing deflection, curvatures, moments, stresses, strains.
               m_convergence_history: Convergence history for Fourier series.
               k_convergence_history: Convergence history for perturbation series.
    """
    results = np.zeros((18, n_nodesY, n_nodesX))
    W, Wxx, Wyy, Wxy, Mx, My, Mxy, Sxx, Syy, Sxy, S1, S2, Svm, Exx, Eyy, Exy, E1, E2 = results[:]
    
    x = np.linspace(0, plate.a, n_nodesX)
    y = np.linspace(0, plate.b, n_nodesY)
    
    Ys = {} 
    
    m_convergence_history = []
    k_convergence_history = []
    
    k = 0
    while k < k_max:
        W_k_contrib = np.zeros((n_nodesY, n_nodesX))
        Wxx_k_contrib = np.zeros((n_nodesY, n_nodesX))
        Wyy_k_contrib = np.zeros((n_nodesY, n_nodesX))
        Wxy_k_contrib = np.zeros((n_nodesY, n_nodesX))
        
        m_history_for_k = []
        
        m = 1
        while m < N_max:
            def calcfkm_k_m():
                fkm = np.zeros((n_nodesY))
                if k == 0 and m % 2 != 0:
                    fkm = 4 * plate.Q / (m * pi * plate.D0) * np.ones((n_nodesY,))
                elif k > 0: 
                    for i in range(1, k + 1):
                        if (k - i, m) in Ys:
                            Y_prev, Yy_prev, Yyy_prev, Yyyy_prev, Yyyyy_prev = Ys[(k - i, m)]
                            
                            Di_val = Di(i, y, plate)
                            Diy_val = Diy(i, y, plate, n_nodesY)
                            Diyy_val = Diyy(i, y, plate, n_nodesY)
                            m_pi_a = m * pi / plate.a
                            
                            A = Di_val * Yyyyy_prev + 2 * Diy_val * Yyyy_prev - 2 * (m_pi_a)**2 * Diy_val * Yy_prev
                            B = (Diyy_val - 2 * Di_val * (m_pi_a)**2) * Yyy_prev
                            C = (Di_val * (m_pi_a)**4 - plate.nu * (m_pi_a)**2 * Diyy_val) * Y_prev
                            
                            if abs(plate.D0) > 1e-12:
                                fkm += -1 / plate.D0 * (A + B + C)
                return fkm

            def dU_dy_k_m(y_bvp, U_bvp):
                current_fkm = calcfkm_k_m()
                if U_bvp.shape[1] != len(current_fkm):
                     fkm_interp = np.interp(y_bvp, y, current_fkm) 
                else:
                     fkm_interp = current_fkm

                m_pi_a = m * pi / plate.a
                return np.vstack([U_bvp[1], U_bvp[2], U_bvp[3], 
                                  fkm_interp + 2*(m_pi_a)**2*U_bvp[2] - (m_pi_a)**4*U_bvp[0]])
            
            def BCs(y0_bvp, yb_bvp):
                return [y0_bvp[0], y0_bvp[2], yb_bvp[0], yb_bvp[2]]    

            Y_guess = np.zeros((4, n_nodesY)) 
            sol = solve_bvp(dU_dy_k_m, BCs, y, Y_guess, max_nodes=n_nodesY*2, tol=1e-4, verbose=0)
            
            if sol.status != 0:
                Y_km, Yy_km, Yyy_km, Yyyy_km = np.zeros((4, n_nodesY))
                Yyyyy_km = np.zeros(n_nodesY)
            else:
                 sol_y = sol.sol(y)
                 Y_km, Yy_km, Yyy_km, Yyyy_km = sol_y
                 fkm_resampled = calcfkm_k_m()
                 m_pi_a = m * pi / plate.a
                 Yyyyy_km = fkm_resampled + 2*(m_pi_a)**2*Yyy_km - (m_pi_a)**4*Y_km

            Ys[(k, m)] = np.array([Y_km, Yy_km, Yyy_km, Yyyy_km, Yyyyy_km])

            m_pi_a = m * pi / plate.a
            sin_term = sin(m_pi_a * x)
            cos_term = cos(m_pi_a * x)
            e_k_term = (plate.e**k) if abs(plate.e) > 1e-12 else (1.0 if k==0 else 0.0)

            dW_km  = np.outer(Y_km, sin_term) * e_k_term
            dWxx_km = np.outer(Y_km, -sin_term) * (m_pi_a**2) * e_k_term
            dWyy_km = np.outer(Yyy_km, sin_term) * e_k_term
            dWxy_km = np.outer(Yy_km, cos_term) * m_pi_a * e_k_term

            W_k_contrib += dW_km
            Wxx_k_contrib += dWxx_km
            Wyy_k_contrib += dWyy_km
            Wxy_k_contrib += dWxy_km

            norm_W_k = np.linalg.norm(W_k_contrib)
            norm_dW_km = np.linalg.norm(dW_km)
            
            rel_err_m = 0.0
            converged_m = False
            if m > 1 and norm_W_k > 1e-12:
                rel_err_m = norm_dW_km / norm_W_k
                if rel_err_m < m_tol:
                    converged_m = True
            elif m == 1 and norm_W_k > 1e-12:
                 rel_err_m = norm_dW_km / norm_W_k

            m_history_for_k.append((m, rel_err_m))

            if converged_m:
                break

            m += 1
            if m >= N_max:
                 print(f"Warning: Fourier series (m) did not converge within N_max={N_max} for k={k}")
        
        m_convergence_history.append(m_history_for_k)

        norm_W = np.linalg.norm(W)
        norm_W_k_contrib = np.linalg.norm(W_k_contrib)

        rel_err_k = 0.0
        converged_k = False
        if k > 0 and norm_W > 1e-12:
            rel_err_k = norm_W_k_contrib / norm_W
            if rel_err_k < k_tol:
                converged_k = True
        elif k == 0 and norm_W_k_contrib > 1e-12:
            rel_err_k = 1.0 
        
        k_convergence_history.append((k, rel_err_k))

        W += W_k_contrib
        Wxx += Wxx_k_contrib
        Wyy += Wyy_k_contrib
        Wxy += Wxy_k_contrib

        if converged_k:
             break

        k += 1
        if k >= k_max:
            print(f"Warning: Perturbation series (k) did not converge within k_max={k_max}")

    D_y = plate.flexural_rigidity(y)
    Mx = -(D_y[:, np.newaxis] * (Wxx + plate.nu * Wyy))
    My = -(D_y[:, np.newaxis] * (Wyy + plate.nu * Wxx))
    Mxy = -(D_y[:, np.newaxis] * (1 - plate.nu) * Wxy)

    z = plate.thickness(y) / 2
    Exx = -(z[:, np.newaxis] * Wxx)
    Eyy = -(z[:, np.newaxis] * Wyy)
    Exy = -2 * (z[:, np.newaxis] * Wxy)
   
    E_term = plate.E / (1 - plate.nu**2)
    G_term = plate.E / (2 * (1 + plate.nu))
    Sxx = E_term * (Exx + plate.nu * Eyy)
    Syy = E_term * (Eyy + plate.nu * Exx)
    Sxy = G_term * Exy
    
    S_avg = (Sxx + Syy) / 2
    S_diff_half = (Sxx - Syy) / 2
    tau_max_sq = S_diff_half**2 + Sxy**2
    tau_max_sq[tau_max_sq < 0] = 0 
    tau_max = np.sqrt(tau_max_sq)
    S1 = S_avg + tau_max
    S2 = S_avg - tau_max
    
    Svm_sq = S1**2 - S1 * S2 + S2**2
    Svm_sq[Svm_sq < 0] = 0
    Svm = np.sqrt(Svm_sq)
    
    E_avg = (Exx + Eyy) / 2
    E_diff_half = (Exx - Eyy) / 2
    gamma_max_sq = E_diff_half**2 + (Exy/2)**2
    gamma_max_sq[gamma_max_sq < 0] = 0
    gamma_max = np.sqrt(gamma_max_sq)
    E1 = E_avg + gamma_max
    E2 = E_avg - gamma_max
   
    results[:] = W, Wxx, Wyy, Wxy, Mx, My, Mxy, Sxx, Syy, Sxy, S1, S2, Svm, Exx, Eyy, Exy, E1, E2
    return results, m_convergence_history, k_convergence_history


titles = [r"$w$", r"$w_{xx}$", r"$w_{yy}$", r"$w_{xy}$",r"$M_{x}$", r"$M_{y}$", r"$M_{xy}$", r"$\sigma_{xx}$",r"$\sigma_{yy}$",
          r"$\sigma_{xy}$",r"$\sigma_{1}$", r"$\sigma_{2}$", r"$\sigma_{VM}$", r"$\epsilon_{xx}$",r"$\epsilon_{yy}$",
          r"$\epsilon_{xy}$",r"$\epsilon_{1}$",r"$\epsilon_{2}$"]

title_map = {title: i for i, title in enumerate(titles)}






