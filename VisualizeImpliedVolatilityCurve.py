# Used libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimize
import math
from gridInterpolation import gridinterpol
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec



def visualize(options_data, symbol):
    strike_data = options_data["Strike"]
    timetomat_data = options_data["TimeToMaturity"]
    implied_vol = options_data["ImpliedVolatility"]

    # Visualization of the implied volatility

    fig = plt.figure(figsize=(20, 20))

    # Gridspec erstellen: 3 Zeilen, 2 Spalten
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 2])

    # Two upper subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Oben links
    ax2 = fig.add_subplot(gs[0, 1])  # Oben rechts
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')

    ax1.scatter(strike_data, implied_vol)
    ax1.set_xlabel('Strike-Price (in €)', fontsize=25)
    ax1.set_ylabel('Implied Volatility', fontsize=25)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.scatter(timetomat_data, implied_vol)
    ax2.set_xlabel('Time to Maturity (in years)', fontsize=25)
    ax2.set_ylabel('Implied Volatility', fontsize=25)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Data points as 3D scatter
    ax3.scatter(strike_data, timetomat_data, implied_vol)
    ax3.set_xlabel('Strike-Price\n(in €)', fontsize=25)
    ax3.set_ylabel('Time to Maturity\n(in years)', fontsize=25)
    ax3.set_zlabel('Implied Volatility', fontsize=25)
    ax3.xaxis.labelpad = 25
    ax3.yaxis.labelpad = 25
    ax3.zaxis.labelpad = 20
    ax3.view_init(20, 130)
    ax3.set_yticks(list(np.arange(0, 2.5, 0.5)))
    x_min, x_max = ax3.get_xlim()
    ax3.set_xlim(x_max, x_min)  # Umkehren
    ax3.tick_params(axis='both', which='major', labelsize=20)

    # Pull interpolated data
    data_interp = gridinterpol(options_data)
    X, Y, Z = data_interp
    X, Y = np.meshgrid(X, Y)

    # Data points as a plane
    norm = LogNorm(vmin=0.2, vmax=1)  # adjust colormap for better differentiation of small values in the plot
    ax4.plot_surface(X, Y, Z.transpose(), cmap="Blues", norm=norm)
    x_min, x_max = ax4.get_xlim()
    ax4.set_xlim(x_max, x_min)
    ax4.view_init(20, 130)
    ax4.set_xlabel('Strike-Price\n(in €)', fontsize=25)
    ax4.set_ylabel('Time to Maturity\n(in years)', fontsize=25)
    ax4.set_zlabel('Implied Volatility', fontsize=25)
    ax4.xaxis.labelpad = 25
    ax4.yaxis.labelpad = 25
    ax4.zaxis.labelpad = 20
    ax4.set_yticks(list(np.arange(0, 2.5, 0.5)))
    ax4.set_xticks(list(np.arange(50, 301, 50)))
    ax4.tick_params(axis='both', which='major', labelsize=20)

    fig.suptitle('Implied Volatility From Traded Call-Options On ' + str(symbol) + "\nas of " + str(
        datetime.datetime.today().strftime('%Y-%m-%d %H:%M')), fontsize=30, y=0.96)
    plt.show()