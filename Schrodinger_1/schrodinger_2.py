import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np


N = 100
x_k = np.linspace(0, 1, N)
delta_tau = 0.0001


@njit(cache=True)
def operator_H(Psi, x_k, tau, omega, array_to_fill):
    kappa = 1
    delta_x = x_k[1] - x_k[0]
    
    for k in range(len(Psi) - 1):
        if k == 0:
            array_to_fill[k] = 0
        else:
            first = (-1 / 2) * (Psi[k + 1] + Psi[k - 1] - 2 * Psi[k]) / (delta_x ** 2)
            second = kappa * (x_k[k] - (1 / 2)) * Psi[k] * np.sin(omega * tau)
            
            array_to_fill[k] = first + second
    
    array_to_fill[len(Psi) - 1] = 0
    return array_to_fill


@njit(cache=True)
def calc_psi_half_R(Psi_R, Psi_I, x_k, tau, omega, delta_tau):
    Psi_half_R = Psi_R + operator_H(Psi=Psi_I, x_k=x_k, tau=tau, omega=omega, array_to_fill=np.empty_like(Psi_R)) * (
                delta_tau / 2)
    return Psi_half_R


@njit(cache=True)
def calc_next_psi_I(Psi_I, Psi_half_R, x_k, tau, omega, delta_tau):
    Psi_next_I = Psi_I - operator_H(Psi=Psi_half_R, x_k=x_k, tau=tau, omega=omega,
                                    array_to_fill=np.empty_like(Psi_half_R)) * delta_tau
    return Psi_next_I


@njit(cache=True)
def calc_next_psi_R(Psi_half_R, Psi_next_I, x_k, tau, omega, delta_tau):
    Psi_next_R = Psi_half_R + operator_H(Psi=Psi_next_I, x_k=x_k, tau=tau, omega=omega,
                                         array_to_fill=np.empty_like(Psi_half_R)) * (delta_tau / 2)
    return Psi_next_R


@njit(cache=True)
def calc_energy(Psi_R, Psi_I, x_k, tau, omega):
    delta_x = x_k[1] - x_k[0]
    
    first = Psi_R * operator_H(Psi=Psi_R, x_k=x_k, tau=tau, omega=omega, array_to_fill=np.empty_like(Psi_R))
    second = Psi_I * operator_H(Psi=Psi_I, x_k=x_k, tau=tau, omega=omega, array_to_fill=np.empty_like(Psi_I))
    
    E = delta_x * np.sum(first + second)
    return E


@njit(cache=True)
def calc_one_sim_step(Psi_R, Psi_I, x_k, tau, omega, delta_tau):
    Psi_half_R = calc_psi_half_R(Psi_R=Psi_R, Psi_I=Psi_I, x_k=x_k, tau=tau, omega=omega, delta_tau=delta_tau)
    Psi_next_I = calc_next_psi_I(Psi_I=Psi_I, Psi_half_R=Psi_half_R, x_k=x_k, tau=tau, omega=omega, delta_tau=delta_tau)
    Psi_next_R = calc_next_psi_R(Psi_half_R=Psi_half_R, Psi_next_I=Psi_next_I, x_k=x_k, tau=tau, omega=omega,
                                 delta_tau=delta_tau)
    
    return Psi_next_R, Psi_next_I


@njit(cache=True)
def run_sim(n, x_k, omega, delta_tau, time_steps):
    tau = 0
    
    Psi_R_init = np.sqrt(2) * np.sin(n * np.pi * x_k)
    Psi_I_init = np.zeros_like(Psi_R_init)
    
    Psi_Rs = np.empty((time_steps, len(Psi_R_init)))
    Psi_Ims = np.empty_like(Psi_Rs)
    Es = np.empty(time_steps)
    
    for step in range(time_steps):
        if step == 0:
            Psi_R, Psi_I = calc_one_sim_step(Psi_R=Psi_R_init, Psi_I=Psi_I_init, x_k=x_k, tau=tau, omega=omega,
                                             delta_tau=delta_tau)
        else:
            Psi_R, Psi_I = calc_one_sim_step(Psi_R=Psi_R, Psi_I=Psi_I, x_k=x_k, tau=tau, omega=omega,
                                             delta_tau=delta_tau)
        tau += delta_tau
        Psi_Rs[step] = Psi_R
        Psi_Ims[step] = Psi_I
        
        E = calc_energy(Psi_R=Psi_R, Psi_I=Psi_I, x_k=x_k, tau=tau, omega=omega)
        Es[step] = E
    
    Psi_norms = np.sum((Psi_Rs ** 2 + Psi_Ims ** 2) * (x_k[1] - x_k[0]), axis=1)
    return Psi_norms, Es


def plot_psi_norms(psi_norms):
    fig, ax = plt.subplots(dpi=100)
    ax.set_title('Norma funkcji falowej w kolejnych krokach czasowych')
    
    ax.plot(psi_norms)
    ax.ticklabel_format(useOffset=False)
    plt.show()


def plot_energy(Es):
    fig, ax = plt.subplots(dpi=100)
    ax.set_title('Energia układu w kolejnych krokach czasowych')
    
    ax.plot(Es)
    ax.ticklabel_format(useOffset=False)
    plt.show()


@njit(cache=True, parallel=True)
def calc_max_energies(n, x_k, omegas, delta_tau, time_steps):
    Es_max = np.empty(len(omegas))
    for i in prange(len(omegas)):
        print(i, ' z ', len(omegas))
        Psi_norms, Es = run_sim(n=n, x_k=x_k, omega=omegas[i], delta_tau=delta_tau, time_steps=time_steps)
        Es_max[i] = np.max(Es)
    
    return Es_max


def unknown_lorentz_func(omega, A, gamma, omega_0):
    return A * ((gamma / 2) / ((omega - omega_0) ** 2 + (gamma / 2) ** 2))


def fit_lorentz(x_data, y_data):
    from scipy.optimize import curve_fit
    
    # Initial guess for the parameters
    initial_guess = [1 / np.pi, 0.05, 1]
    # Perform the curve-fit
    try:
        popt, pcov = curve_fit(unknown_lorentz_func, x_data, y_data, initial_guess)
    except ValueError("Fitting problem, returns nan"):
        popt = (np.nan, np.nan, np.nan)
    
    return popt


def plot_peak_and_fit(omegas, Es, params):
    fig, ax = plt.subplots(dpi=100)
    ax.set_title('Energia od omegi')
    ax.set_xlabel(r'$\frac{\omega}{\omega_0}$')
    
    y_vals = unknown_lorentz_func(omegas, *params)
    
    import pandas as pd
    df = pd.DataFrame(np.array([omegas, y_vals]).T)
    df.sort_values(df.columns[0], inplace=True)
    
    ax.scatter(omegas, Es)
    ax.plot(df[0], df[1])
    ax.ticklabel_format(useOffset=False)
    plt.show()


def plot_normalized_peak_and_fit(omegas, Es, params):
    fig, ax = plt.subplots(dpi=100)
    ax.set_title('Energia (przesunięta do 0) od omegi')
    ax.set_xlabel(r'$\frac{\omega}{\omega_0}$')
    
    y_vals = unknown_lorentz_func(omegas, *params)
    
    import pandas as pd
    df = pd.DataFrame(np.array([omegas, y_vals]).T)
    df.sort_values(df.columns[0], inplace=True)
    
    ax.scatter(omegas, Es - Es[0])
    ax.plot(df[0], df[1])
    ax.ticklabel_format(useOffset=False)
    plt.show()


"""# main"""

omega_0 = (3 * np.pi ** 2) / 2
points = 10
start = 0.8
omegas = np.append(np.linspace(start, 1 + (1 - start), points), 1) * omega_0
omegas = np.append(omegas, np.linspace(0.95, 1.05, points) * omega_0)
omegas = np.append(omegas, np.linspace(0.98, 1.02, points) * omega_0)

Es = calc_max_energies(n=1, x_k=x_k, omegas=omegas, delta_tau=delta_tau, time_steps=int(4e5))

params = fit_lorentz(omegas / omega_0, Es)
plot_peak_and_fit(omegas=omegas / omega_0, Es=Es, params=params)

params = fit_lorentz(omegas, Es - Es[0])
plot_normalized_peak_and_fit(omegas=omegas / omega_0, Es=Es, params=params)

# %timeit Psi_norms, Ets = run_sim(n=1, x_k=x_k, omega=omega_0, delta_tau=delta_tau, time_steps=int(5e5))
# plot_energy(Es=Ets)
# plot_psi_norms(psi_norms=Psi_norms)