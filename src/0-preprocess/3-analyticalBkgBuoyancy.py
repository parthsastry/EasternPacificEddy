'''
Script to fit a piecewise function to the buoyancy profile given by the
integration of the smoothed, averaged N^2 profile.
The N^2 profile is an average of three casts (S37, S38, S39) from SR2114
(PI: Mark Altabet)

Author: Parth Sastry
'''

import numpy as np
from scipy.optimize import minimize
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import xarray as xr


def piecewise_func(x, params):
    a_tanh, b_tanh, c_tanh, d_tanh, a_log, b_log, c_log, cutoff = params
    if x < cutoff:
        return a_log*np.log(-(x + b_log)) + c_log
    else:
        return a_tanh*np.tanh(b_tanh*(x + c_tanh)) + d_tanh


def objective(params, x, y):
    return np.sum(
        (y - np.array([piecewise_func(xi, params) for xi in x]))**2
    )


def continuity_constraint(params):
    a_tanh, b_tanh, c_tanh, d_tanh, a_log, b_log, c_log, cutoff = params
    return (
        a_tanh*np.tanh(b_tanh*(cutoff + c_tanh)) +
        d_tanh - a_log*np.log(-(cutoff + b_log)) - c_log
    )


def derivative_constraint(params):
    a_tanh, b_tanh, c_tanh, d_tanh, a_log, b_log, c_log, cutoff = params
    return (
        a_tanh*b_tanh*(1-np.tanh(b_tanh*(cutoff + c_tanh))**2) -
        a_log/(cutoff + b_log)
    )


def surface_buoyancy(params):
    a_tanh, b_tanh, c_tanh, d_tanh, a_log, b_log, c_log, cutoff = params
    return a_tanh*np.tanh(b_tanh*c_tanh) + d_tanh + 9.8065


# Load Data
buoyancy_mat = sio.loadmat('../../data/processed/smoothedN2_CTD_S37-39.mat')
depth = buoyancy_mat['depth'].flatten()
buoyancy = buoyancy_mat['buoyancy'].flatten()

initial_params = [
    0.03242, 0.03152, 117.9, -9.838, -0.007545, 120, -9.821, -143
]
constraints = [
    {'type': 'eq', 'fun': continuity_constraint},
    {'type': 'eq', 'fun': derivative_constraint},
    {'type': 'eq', 'fun': surface_buoyancy}
]

result = minimize(
    objective, initial_params, args=(depth, buoyancy),
    constraints=constraints
)
optimized_params = result.x

# Save to netCDF4 file
fitParams = xr.DataArray(
    optimized_params.reshape((2, 4))
)
fitParams.to_netcdf('../../data/processed/buoyancyFitParams.nc')

# Show profile
profileFigure = False

if profileFigure:

    CTD_data = sio.loadmat('../../data/processed/N2_CTD_S37-40.mat')
    s37_dens = CTD_data['s37_mini'][0][0][0]
    s37_depth = -1*np.array(CTD_data['s37_mini'][0][0][3]).astype(float)

    s38_dens = CTD_data['s38_mini'][0][0][0]
    s38_depth = -1*np.array(CTD_data['s38_mini'][0][0][3]).astype(float)

    s39_dens = CTD_data['s39_mini'][0][0][0]
    s39_depth = -1*np.array(CTD_data['s39_mini'][0][0][3]).astype(float)

    s37_b = -9.807*s37_dens/1020.5
    s38_b = -9.807*s38_dens/1020.5
    s39_b = -9.807*s39_dens/1020.5

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        s37_depth, s37_b, s=2, label='S37 Buoyancy Data', marker='o'
    )
    ax.scatter(
        s38_depth, s38_b, s=2, label='S38 Buoyancy Data', marker='o'
    )
    ax.scatter(
        s39_depth, s39_b, s=2, label='S39 Buoyancy Data', marker='o'
    )
    ax.scatter(
        depth, buoyancy, s=2,
        label=r'Integrated Buoyancy Data from smoothed $N^2$',
        marker='x'
    )
    ax.plot(
        depth, [piecewise_func(x, optimized_params) for x in depth],
        'k', label='Analytical Buoyancy Profile'
    )

    ax.set_xlabel(r'Depth (m)')
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(20))

    ax.set_ylabel(r'Buoyancy ($ms^{-2}$)')
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))

    ax.set_title("Background Buoyancy Profile")
    ax.legend()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.savefig('../../output/figures/buoyancyFit.png')
