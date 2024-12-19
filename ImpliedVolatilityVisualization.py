# Used libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimize
import math

pd.set_option('display.max_columns', None)  # Zeigt alle Spalten an


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

        implied_volatility = -1

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
    strike_data = options_data["Strike"]
    timetomat_data = options_data["TimeToMaturity"]
    implied_vol = options_data["ImpliedVolatility"]

    # Visualization of the implied volatility
    fig = plt.figure(figsize=(20, 20))
    import matplotlib.gridspec as gridspec

    # Gridspec erstellen: 3 Zeilen, 2 Spalten
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 3])
    #gs = gridspec.GridSpec(1, 2)

    # Die oberen beiden 2D-Subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Oben links
    ax2 = fig.add_subplot(gs[0, 1])  # Oben rechts
    ax3 = fig.add_subplot(gs[1:, :], projection='3d')

    ax1.scatter(strike_data, implied_vol)
    ax1.set_xlabel('Strike-Price (in €)', fontsize=20)
    ax1.set_ylabel('Implied Volatility', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ax2.scatter(timetomat_data,implied_vol )
    ax2.set_xlabel('Time to Maturity (in years)', fontsize=20)
    ax2.set_ylabel('Implied Volatility', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=15)


    ax3.scatter(strike_data,timetomat_data,implied_vol)
    ax3.set_xlabel('Strike-Price\n(in €)', fontsize=20)
    ax3.set_ylabel('Time to Maturity\n(in years)', fontsize=20)
    ax3.set_zlabel('Implied Volatility', fontsize=20)
    ax3.xaxis.labelpad = 20
    ax3.yaxis.labelpad = 20
    ax3.view_init(20, 100)
    ax3.set_yticks(list(np.arange(0, 2.5, 0.5)))
    ax3.tick_params(axis='both', which='major', labelsize=15)


    fig.suptitle('Implied Volatility From Traded Call-Options On ' + str(symbol) + "\nas of " + str(datetime.datetime.today().strftime('%Y-%m-%d %H:%M')), fontsize=30, y=0.96)
    plt.show()


# Setting parameters:
symbol = "GOOG"                 # Symbol to look at
current_asset_price = 185      # Current stock (asset) price
risk_free_rate = 0.03           # Risk-free interest rate
min_TimeToMaturity = 0     # Minimum time to maturity to take into data

# Call above defined function for calculation and visualization
ImpliedVolatilityCallVisualization(symbol, current_asset_price, risk_free_rate, min_TimeToMaturity)

