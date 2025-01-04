# Used libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimize
import math
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp2d



def gridinterpol(data):

    data = pd.DataFrame(data)
    data = data.sort_values(by=["Strike", "TimeToMaturity"], ascending=True)

    dim1_max = data["Strike"].max()
    dim1_min = data["Strike"].min()
    dim2_max = data["TimeToMaturity"].max()
    dim2_min = data["TimeToMaturity"].min()

    strike_un = data["Strike"].unique()
    ttm_un = data["TimeToMaturity"].unique()

    strikes = np.arange(dim1_min, dim1_max, 5)

    M = np.zeros((len(strikes), len(ttm_un)), dtype=float, order='C')


    for i in range(len(strikes)):
        for j in range(len(ttm_un)):
            result = data.loc[(data['Strike'] == strikes[i]) & (data['TimeToMaturity'] == ttm_un[j]), 'ImpliedVolatility'].values
            if result != np.nan:
                M[i,j] = result
            else:
                M[i,j] = np.nan


    #### Interpolation
    # Positions of real values and nans
    x, y = np.indices(M.shape)
    valid = ~np.isnan(M)
    points = np.column_stack((x[valid], y[valid]))
    values = M[valid]

    # Interpolation for points
    M = griddata(points, values, (x, y), method='linear')

    #M = M.reshape((len(strikes) * len(ttm_un)))

    return strikes, ttm_un, M

