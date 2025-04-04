import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from cmocean import cm


def plotSlicePV(depthSlice, outdir):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        rf'PV @ z = {round(float(depthSlice.zC), 2)} m' +
        rf'(t = {float(depthSlice.time[0])/(3600*1E9)} h)'
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
            rf'PV @ z = {round(float(depthSlice.zC), 2)} m' +
            rf'(t = {float(depthSlice.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(np.swapaxes(depthSlice.Q[i, :, :], 0, 1))

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(depthSlice.time)
    )
    anim.save(
        outdir + 'depthSlice_' +
        f'{round(float(depthSlice.zC), 2)}' + '_PV.gif'
    )


def plotSliceBuoyancy(depthSlice, outdir):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        rf'Buoyancy @ z = {round(float(depthSlice.zC), 2)} m' +
        rf'(t = {float(depthSlice.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

    cax = ax.pcolormesh(
        depthSlice.xC/1000, depthSlice.yC/1000,
        np.swapaxes(depthSlice.b[0, :, :], 0, 1),
        cmap=cm.balance
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(
        r'$b = -g \frac{\rho}{\rho_0}$'
    )

    def animate(i):
        ax.set_title(
            rf'Buoyancy @ z = {round(float(depthSlice.zC), 2)} m' +
            rf'(t = {float(depthSlice.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(np.swapaxes(depthSlice.b[i, :, :], 0, 1))

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(depthSlice.time)
    )
    anim.save(
        outdir + 'depthSlice_' +
        f'{round(float(depthSlice.zC), 2)}' + '_b.gif'
    )


def plotProfilePV(profile, outdir):

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        f'Vorticity Profile (t = {float(profile.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('Depth [m]')
    cax = ax.pcolormesh(
        profile.xC[64:193]/1000, profile.zC[30:], profile.Q[0, 30:, 64:193],
        cmap=cm.balance
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
    anim.save(outdir + 'PVprofile.gif')


def plotParticleTraj(particleDataset, outdir):

    for i in range(1, 22):
        particleTraj = particleDataset.sel(particle_id=float(i))
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].scatter(particleTraj.x/1E3, particleTraj.y/1E3, color='red')
        axs[0].set_xlabel('x [km]')
        axs[0].set_ylabel('y [km]')
        axs[0].set_xlim(-150, 150)
        axs[0].set_ylim(-150, 150)
        axs[0].set_title(f'Horizontal Trajectory of Particle {int(i)}')

        axs[1].scatter(particleTraj.x/1E3, particleTraj.z, color='green')
        axs[1].set_xlabel('x [km]')
        axs[1].set_ylabel('z [m]')
        axs[1].set_xlim(-150, 150)
        axs[1].set_title(f'Vertical Trajectory of Particle {int(i)}')

        plt.savefig(outdir + f'particleTrajs/particle{int(i)}.png')


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


def plotAnomalyProfile(profile, outdir):

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
    anim.save(outdir + 'buoyancyAnomaly.gif')


def plotAnomalySlice(depthSlice, outdir):

    buoyancyBackground = piecewise_func(float(depthSlice.zC))
    buoyancyAnomaly = depthSlice.b - buoyancyBackground

    fig, ax = plt.subplots()
    ax.set_title(
        rf'Buoyancy Anomaly @ z = {round(float(depthSlice.zC), 2)} m' +
        rf'(t = {float(buoyancyAnomaly.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

    cax = ax.pcolormesh(
        depthSlice.xC/1000, depthSlice.yC/1000,
        np.swapaxes(buoyancyAnomaly[0, :, :], 0, 1),
        cmap=cm.balance
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(
        r'$b_{\text{a}} = -g \frac{\rho}{\rho_0} - b_{\text{bkg}}$'
    )

    def animate(i):
        ax.set_title(
            rf'Buoyancy Anomaly @ z = {round(float(depthSlice.zC), 2)} m' +
            rf'(t = {float(buoyancyAnomaly.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(np.swapaxes(buoyancyAnomaly.b[i, :, :], 0, 1))

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(depthSlice.time)
    )
    anim.save(
        outdir + 'depthSlice_' +
        f'{round(float(depthSlice.zC), 2)}' + '_banom.gif'
    )


def plotOxygenProfile(profile, outdir):

    oxygenProfile = profile.O2
    fig, ax = plt.subplots()
    ax.set_title(
        'Oxygen Concentration Profile ' +
        f'(t = {float(oxygenProfile.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('Depth [m]')
    cax = ax.pcolormesh(
        oxygenProfile.xC/1000, oxygenProfile.zC, oxygenProfile[0, :, :],
        cmap=cm.balance
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(r'Oxygen Concentration')

    def animate(i):
        ax.set_title(
            'Oxygen Concentration Profile ' +
            f'(t = {float(oxygenProfile.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(oxygenProfile[i, :, :])

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(profile.time)
    )
    anim.save(outdir + 'oxygenProfile.gif')
    
    
def plotOxygenSlice(depthSlice, outdir):
    
    oxygenSlice = depthSlice.O2
    fig, ax = plt.subplots()
    ax.set_title(
        rf'Oxygen Concentration Profile  @ z = {round(float(depthSlice.zC), 2)} m' +
        f' (t = {float(oxygenSlice.time[0])/(3600*1E9)} h)'
    )
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')

    cax = ax.pcolormesh(
        depthSlice.xC/1000, depthSlice.yC/1000,
        np.swapaxes(oxygenSlice[0, :, :], 0, 1),
        cmap=cm.balance
    )
    cbar = fig.colorbar(cax)
    cbar.set_label(r'Oxygen Concentration')

    def animate(i):
        ax.set_title(
            rf'Oxygen Concentration Profile  @ z = {round(float(depthSlice.zC), 2)} m' +
            f' (t = {float(oxygenSlice.time[i])/(3600*1E9)} h)'
        )
        cax.set_array(np.swapaxes(oxygenSlice[i, :, :], 0, 1))

    anim = FuncAnimation(
        fig, animate, interval=70, frames=len(depthSlice.time)
    )
    anim.save(
        outdir + 'depthSlice_' +
        f'{round(float(depthSlice.zC), 2)}' + '_O2.gif'
    )
