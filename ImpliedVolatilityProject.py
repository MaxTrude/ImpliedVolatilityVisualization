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
from VisualizeImpliedVolatilityCurve import visualize

pd.set_option('display.max_columns', None)  # Shows all columns when printing data frames


def ImpliedVolatilityCallVisualization(symbol, current_asset_price, risk_free_rate, min_TimeToMaturity):

    tick = yf.Ticker(symbol) # Defining ticker in yFinance-syntax

    # Black-Scholes Formula for valuation of european call-options as defined in financial literature
    def black_scholes_call(S, K, T, r, sigma):

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        # Normal distributions
        N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

        # Valuation of option with passed specifics
        call_price = S * N_d1 - K * math.exp(-r * T) * N_d2
        return call_price


    # Function for calculating implied volatility
    def implied_volatility_call(S, K, T, r, market_price):

        # Target-function: Difference between market price of the option and the Black-Scholes-Valuation
        def objective_function(sigma):
            return black_scholes_call(S, K, T, r, sigma) - market_price

        # Define some possible starting values for the Newton-Algorithm
        X0 = np.arange(0.1, 2, 0.1)


        # Iterate starting values, check if newton converges, save solution if it is calculated
        for x0 in X0:
            try:
                implied_vol = optimize.newton(objective_function, x0, tol=1e-8)
            except RuntimeError:
                if x0 == X0[-1]: implied_vol = -1
                else: continue
            if implied_vol != -1:
                break

        return implied_vol

    options_data = []

    # Iterate over all available maturity dates
    for dates in tick.options:

        Options = tick.option_chain(date=dates)     # Get options data of certain maturity date

        # Calculation block
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None):

            # Get available option-configurations on specific maturity date
            strikes = Options.calls.strike
            last_prices = Options.calls.lastPrice

            # Transform maturity dates into remaining time to maturity in years
            TimeToMat = round((datetime.datetime.strptime(dates, '%Y-%m-%d').date()-datetime.date.today()).days/365,2)
            TimeToMat_Vek = len(strikes)*[TimeToMat]

            # Filter too short remaining times to maturity and add option configuration to "options_data"-list
            if TimeToMat >= min_TimeToMaturity:
                for i in range(len(TimeToMat_Vek)):
                    impVol = round(implied_volatility_call(current_asset_price, strikes[i], TimeToMat_Vek[i], risk_free_rate, last_prices[i]),2)
                    options_data.append([strikes[i], TimeToMat_Vek[i], impVol])

    # Filter-Function to cut out calculation errors (e.g. Newton did not converge)
    fil = lambda x: x[2] != -1
    options_data = filter(fil, options_data)

    # Transform in DataFrame
    options_data = pd.DataFrame(options_data, columns=['Strike', 'TimeToMaturity', 'ImpliedVolatility'])

    # Save calculations in CSV file
    options_data.to_csv("options_data.csv", index=False)
    # Extract column vectors from dataframe

    visualize(options_data, symbol)


# Setting parameters:
symbol = "GOOG"                 # Symbol to look at
current_asset_price = 190      # Current stock (asset) price
risk_free_rate = 0.03           # Risk-free interest rate
min_TimeToMaturity = 1/6          # Minimum time to maturity to take into data

# Call above defined function for calculation and visualization
ImpliedVolatilityCallVisualization(symbol, current_asset_price, risk_free_rate, min_TimeToMaturity)

