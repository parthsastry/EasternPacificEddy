import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cmocean import cm


def plotDepthPV(depthSlice):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        rf'PV @ z = -200 m (t = {float(depthSlice.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

    cax = ax.pcolormesh(
        depthSlice.xC/1000, depthSlice.yC/1000,
        np.swapaxes(depthSlice.Q[0, :, :], 0, 1),
        cmap=cm.balance
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(
        r'$Q = \left[ \zeta + f \hat{z} \right] \cdot$' +
        r'$\left( \frac{\nabla b + N^2 \hat{z}}{fN^2} \right) - 1$'
    )

    def animate(i):
        ax.set_title(
            rf'PV @ z = -200 m (t = {float(depthSlice.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(np.swapaxes(depthSlice.Q[i, :, :], 0, 1))

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(depthSlice.time)
    )
    anim.save('depthSlicePV.gif')


def plotProfilepV(profile):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        f'Vorticity Profile (t = {float(profile.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('Depth [m]')
    cax = ax.pcolormesh(
        profile.xC[64:193]/1000, profile.zC[30:], profile.Q[0, 30:, 64:193],
        cmap=cm.balance,
        vmin=-0.26, vmax=0.1
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(
        r'$Q = \left[ \zeta + f \hat{z} \right] \cdot$' +
        r'$\left( \frac{\nabla b + N^2 \hat{z}}{fN^2} \right) - 1$'
    )

    def animate(i):
        ax.set_title(
            f'Vorticity Profile (t = {float(profile.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(profile.Q[i, 30:, 64:193])

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(profile.time)
    )
    anim.save('PVprofile.gif')


def plotParticleTraj(particleDataset):
    particleTrajs = [
        particleDataset.sel(particle_id=float(i)) for i in range(1, 22)
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('Depth [m]')
    for i in range(21):
        ax.plot(
            particleTrajs[i].x, particleTrajs[i].y, particleTrajs[i].z,
            marker='.', label=f'Particle {i}'
        )
    # ax.scatter(
    #   particle1Traj.x, particle1Traj.y, particle1Traj.z, color='red'
    # )
    ax.legend()
    plt.savefig('particleTraj.png')


analyticalB_params = xr.open_dataset(
    "../../data/processed/buoyancyFitParams.nc"
).__xarray_dataarray_variable__
cutoff = analyticalB_params[1, 3]
a_tanh, b_tanh, c_tanh, d_tanh = analyticalB_params[0, :]
a_log, b_log, c_log = analyticalB_params[1, :3]


def piecewise_func(x):
    if x < cutoff:
        return a_log*np.log(-(x + b_log)) + c_log
    else:
        return a_tanh*np.tanh(b_tanh*(x + c_tanh)) + d_tanh


def piecewise_deriv(x):
    if x < cutoff:
        return a_log/(x + b_log)
    else:
        return a_tanh*b_tanh*(1-np.tanh(b_tanh*(x + c_tanh))**2)


def plotAnomalyProfile(profile):

    buoyancyBackground = np.array([piecewise_func(z) for z in profile.zC])
    # N2Background = np.array([piecewise_deriv(z) for z in profile.zC])
    buoyancyAnomaly = profile.b - buoyancyBackground[
        np.newaxis, ..., np.newaxis
    ]

    fig, ax = plt.subplots()
    ax.set_title(
        f'Buoyancy Anomaly (t = {float(buoyancyAnomaly.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('Depth [m]')
    cax = ax.pcolormesh(
        buoyancyAnomaly.xC/1000, buoyancyAnomaly.zC, buoyancyAnomaly[0, :, :],
        cmap=cm.balance, vmin=np.min(buoyancyAnomaly[-1, :, :]),
        vmax=np.max(buoyancyAnomaly[-1, :, :])
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(r'Buoyancy Anomaly [ms$^{-2}$]')

    def animate(i):
        ax.set_title(
            'Buoyancy Anomaly ' +
            f'(t = {float(buoyancyAnomaly.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(buoyancyAnomaly[i, :, :])

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(profile.time)
    )
    anim.save('buoyancyAnomaly.gif')
