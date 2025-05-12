# ANALISIS DEL EUROPIO 152

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

data = pd.read_csv("Eu-152_15min_NaI-Tl_bicron900V.csv", header=None, names=["Canal", "Conteo"])

# Conteos vs. canal
plt.figure(figsize=(12, 5))
plt.semilogy(data["Canal"], data["Conteo"], color='navy')
plt.title("Espectro de 152Eu (Canal vs Conteo)")
plt.xlabel("Canal")
plt.ylabel("Conteo")
plt.grid(True)
plt.tight_layout()
plt.savefig("espectro_canal_vs_conteo.png")
plt.show()

# Picos de energía
canales_cal = np.array([78, 148, 207, 450, 564, 637, 802]) 
energias_cal = np.array([122, 245, 344, 779, 964, 1112, 1408])  

# Ajuste lineal
slope, intercept, r_value, _, _ = linregress(canales_cal, energias_cal)
print(f"\nCalibración obtenida: Energía = {slope:.4f} * Canal + {intercept:.2f}")
print(f"Coeficiente de determinación R² = {r_value**2:.4f}")

# Conteos vs. energía
canal_fit = np.linspace(min(canales_cal) - 10, max(canales_cal) + 10, 500)
energia_fit = slope * canal_fit + intercept

plt.figure(figsize=(8, 5))
plt.scatter(canales_cal, energias_cal, color='red', label='Puntos de calibración', zorder=3)
plt.plot(canal_fit, energia_fit, color='blue', label=f'Regresión lineal\nE = {slope:.2f}·Canal + {intercept:.2f}')
plt.title("Calibración Energética (Canal vs Energía)")
plt.xlabel("Canal")
plt.ylabel("Energía (keV)")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("calibracion_lineal.png")
plt.show()


data["Energía (keV)"] = slope * data["Canal"] + intercept


plt.figure(figsize=(12, 5))
plt.semilogy(data["Energía (keV)"], data["Conteo"], color='darkgreen')
plt.title("Espectro de 152Eu (Energía vs Conteo)")
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo")
plt.grid(True)
plt.tight_layout()
plt.savefig("espectro_energia_vs_conteo.png")
plt.show()

# ANALISIS DEL CESIO 137

data_137cs = pd.read_csv("Cs-137_15min_NaI-Tl_bicron_900V.csv", header=None, names=["Canal", "Conteo"])
data_137cs["Energía (keV)"] = slope * data_137cs["Canal"] + intercept

plt.figure(figsize=(10,6))
plt.semilogy(data_137cs["Energía (keV)"], data_137cs["Conteo"])
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo")
plt.title("Espectro de 137Cs")
plt.grid(True)
plt.show()


from scipy.optimize import curve_fit

# Datos del pico del Cs-137 (ajusta estos límites según tu espectro)
energy_min = 620  # keV
energy_max = 760  # keV
mask = (data_137cs["Energía (keV)"] >= energy_min) & (data_137cs["Energía (keV)"] <= energy_max)
x_data = data_137cs[mask]["Energía (keV)"].values
y_data = data_137cs[mask]["Conteo"].values

# Función gaussiana
def gauss(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

# Estimación inicial de parámetros (A, mu, sigma, offset)
mu_guess = 662  # Energía esperada del pico (keV)
A_guess = y_data.max() - y_data.min()
sigma_guess = 10  # Estimación inicial del ancho (keV)
offset_guess = y_data.min()

# Ajuste de la gaussiana
popt, pcov = curve_fit(gauss, x_data, y_data, p0=[A_guess, mu_guess, sigma_guess, offset_guess])
A_fit, mu_fit, sigma_fit, offset_fit = popt

# Predicciones del modelo
y_pred = gauss(x_data, *popt)

# Cálculo del R²
residuals = y_data - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"Coeficiente de determinación R² = {r_squared:.4f}")

# Cálculo del FWHM
fwhm = 2.355 * sigma_fit  # FWHM = 2.355 * sigma
print(f"\nResultados del ajuste gaussiano:")
print(f" - Centro del pico (μ): {mu_fit:.2f} keV")
print(f" - Desviación estándar (σ): {sigma_fit:.2f} keV")
print(f" - FWHM: {fwhm:.2f} keV")

# Gráfico del ajuste gaussiano
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'b-', label="Datos originales")
plt.plot(x_data, gauss(x_data, *popt), 'r--', label=f"Ajuste gaussiano\nμ = {mu_fit:.2f} keV\nFWHM = {fwhm:.2f} keV")
plt.xlabel("Energía (keV)")
plt.ylabel("Conteo")
plt.title("Ajuste gaussiano al pico del espectro de $^{137}$Cs")
plt.legend()
plt.grid(True)
plt.show()
