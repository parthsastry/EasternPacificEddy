using Pkg
Pkg.activate(".")

Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Measures
using CairoMakie
#using GibbsSeaWater # only necessary if using T and S for equation of state
using Statistics
using Printf
using NCDatasets
using CUDA
using Adapt

# sw_SurfaceVortex = true # surface (true) or subsurface (false) vortex
sw_NonUniformGrid = true; # non-uniform (true) or uniform (false) z-grid
sw_UseSpongeLayer = true; # damping circular sponge layer (true) or not (false)

# ============================= #
#        Grid Parameters        #
# ============================= #

const Nx = 256; # x grid points
const Ny = 256; # y grid points
const Nz = 64;  # z grid points

const L = 150kilometers; # Half-eddy length scale
const H = 500meters;     # Half-eddy depth scale

# NOTE - WILL NEED TO INCREASE FOR PROD RUNS
const Lx = 15*L;      # zonal domain width
const Ly = 15*L;      # meridional domain width
const Lz = 7.5*H;      # vertical domain height

# =============================== #
#           Grid Setup            #
# =============================== #

dx = Lx / Nx;   # zonal grid spacing
dy = Ly / Ny;   # meridional grid spacing
# NOTE - this is some sort of average grid spacing in the case of non-uniform z-grid spacing
dz = Lz / Nz;   # vertical grid spacing

# NOTE - this is some sort of average aspect ratio in the case of non-uniform z-grid spacing
δ = dx/dz;      # aspect ratio

println("dx = ", string(dx), " m");

xGrid = (-Lx/2, Lx/2);
yGrid = (-Ly/2, Ly/2);

if sw_NonUniformGrid
    println("Using a non-uniform hyperbolically stretched z-grid")
    # Hyperbolically Stretched near surface
    σ = 2.5; # stretching parameter
    hyperbolically_spaced_faces(k) = - Lz * (1 - tanh(σ * (k - 1) / Nz) / tanh(σ));
    zGrid = hyperbolically_spaced_faces;
else
    println("Using a uniform z-grid")
    zGrid = (-Lz, 0.0);
end

grid = RectilinearGrid(
    GPU(),
    size = (Nx, Ny, Nz),
    x = xGrid, y = yGrid, z = zGrid,
    halo = (3, 3, 3),
    topology = (
        Oceananigans.Grids.Periodic, Oceananigans.Grids.Periodic, Oceananigans.Grids.Bounded
    )
);

xᶜ = CuArray([-(Lx/2 + dx/2) + i*dx for i in 1:Nx]);
yᶜ = CuArray([-(Ly/2 + dy/2) + i*dy for i in 1:Ny]);
zᶜ = CuArray([mean((hyperbolically_spaced_faces(i), hyperbolically_spaced_faces(i+1))) for i in 1:Nz]);

Xp = CUDA.ones(Nx, Ny, Nz).*xᶜ;
Yp = CUDA.ones(Nx, Ny, Nz).*reshape(yᶜ, (1, Ny));
Zp = CUDA.ones(Nx, Ny, Nz).*reshape(zᶜ, (1, 1, Nz));

# =============================== #
#        Eddy parameters          #
# =============================== #

const α = 2.0;                     # Eddy decay exponent (\alpha = 2 => Gaussian)
# Eddy amplitude at center (roughly maximum possible for non-negative center surface buoyancy)
# NOTE - important parameter in determining strength of eddy - also important for stratification stability
# N̄² - 2gη/H² > 0 for stable stratification in center
const η = 0.2meters;

const g = 9.807meters/second^2;     # gravity
const ρ₀ = 1020.5;                 # kg m⁻³ reference surface density - taken from CTD data

const p₀ = ρ₀*g*η;                 # reference pressure

coriolis = FPlane(latitude=15);
const f₀ = coriolis.f;

# =============================== #
#        Initial conditions       #
# =============================== #

const Zscale = H;

Znorm = Zp ./ Zscale;
Rnorm = sqrt.(Xp.^2 + Yp.^2) ./ L;

χ = exp.((-Rnorm.^α) .+ (-Znorm.^α)); # 3D Gaussian
p = p₀ .* χ;                          # pressure field
ϕ = angle.(Xp .+ im*Yp);              # azimuthal angle

# Mahdinia vortex

Vᵩ = f₀ .* Rnorm .* L ./ 2 .* (-1 .+ sqrt.(1 .- 4 .* α .* p ./(ρ₀ .* f₀.^2 .* L^2))); # azimuthal velocity
b_anom = -α .* p .* Znorm.^α ./ (ρ₀ .* Zp); # buoyancy anomaly field

Vₓ = -real.(Vᵩ .* sin.(ϕ));
Vᵧ = real.(Vᵩ .* cos.(ϕ))

# Initialize velocity fields (can add background flow here if needed)
U = copy(Vₓ);
V = copy(Vᵧ);
W = CUDA.zeros(Nx, Ny, Nz);

# =============================================== #
#    Background Buoyancy/Stratification Field     #
# =============================================== #

param_ds = Dataset("../../data/processed/buoyancyFitParams.nc");
fitParams = param_ds["__xarray_dataarray_variable__"];

const a_tanh = fitParams[1,1]
const b_tanh = fitParams[2,1]
const c_tanh = fitParams[3,1]
const d_tanh = fitParams[4,1]
const a_log = fitParams[1,2]
const b_log = fitParams[2,2]
const c_log = fitParams[3,2]
const cutoff = fitParams[4,2]

function bkgBuoyancy(x, y, z)
    out = (z .< cutoff) .* (a_log .* log.(-(z .+ b_log)) .+ c_log) .+ (z .>= cutoff) .* (a_tanh .* tanh.(b_tanh .* (z .+ c_tanh)) .+ d_tanh)
    out
end

b_bkg = bkgBuoyancy(Xp, Yp, Zp);
bot_N² = @CUDA.allowscalar(a_log / Zp[1,1,1])

b_tot = b_anom .+ b_bkg;

# ================================ #
#       Boundary Conditions        #
# ================================ #

# Buoyancy

const Qflux = 0.0;              # W m⁻² heat flux
const Cₚ = 3991.0;              # J kg⁻¹ K⁻¹ specific heat capacity of seawater
const EminusP = 0.0;            # mm day⁻¹ evaporation minus precipitation

# Temperature Salinity at surface (taken from CTD data)

const S₀ = 33.2801;             # psu surface salinity
const αₜ = 3.26e-4;             # K⁻¹ thermal expansion coefficient
const βₛ = 7.172e-4;            # psu⁻¹ haline contraction coefficient

# Surface Buoyancy Flux (Heating and Precipitation)

const Bflux_heat = Qflux / (ρ₀ * Cₚ) * (αₜ * g);  # m² s⁻³
const Bflux_evap = EminusP *  (g * βₛ) * S₀;      # m² s⁻³

const Bflux_tot = Bflux_heat + Bflux_evap;

B_BCS = FieldBoundaryConditions(
    top = FluxBoundaryCondition(Bflux_tot),
    bottom = GradientBoundaryCondition(bot_N²)
);

# Velocity

const U₁₀ = 10.0meters/second;        # m s⁻² Wind Speed 10 meters above sea level
const θwind = 270;                     # Wind (to) direction (degrees), Clockwise from 0ᵒ - True North
const u₁₀ = U₁₀ * cosd(90 - θwind);    # zonal wind component
const v₁₀ = U₁₀ * sind(90 - θwind);    # meridional wind component

const C_D = 1.5e-3;                    # Drag coefficient
const ρₐ = 1.225;                      # kg m⁻³ air density

const τₓ = C_D * ρₐ * u₁₀ * abs(u₁₀);  # zonal wind stress
const τᵧ = C_D * ρₐ * v₁₀ * abs(v₁₀);  # meridional wind stress

U_BCS = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τₓ/ρ₀)
);
V_BCS = FieldBoundaryConditions(
    top = FluxBoundaryCondition(τᵧ/ρ₀)
);

# Oxygen Tracer Boundary Conditions (future implementation could involve steady subsurface oxygen supply)

O2_BCS = FieldBoundaryConditions(
    top = FluxBoundaryCondition(0.0)
);

# ================================= #
#         Turbulent Closure         #
# ================================= #

# Anisotropic Diffusivity

const κₕ = 2.0;            # m² s⁻¹ horizontal diffusivity
const νₕ = 2.0;            # m² s⁻¹ horizontal viscosity
const κᵥ = (1.0/δ) * κₕ * 0.001;  # m² s⁻¹ vertical diffusivity
const νᵥ = (1.0/δ) * νₕ * 0.001;  # m² s⁻¹ vertical viscosity

horizontal_diff_closure = HorizontalScalarDiffusivity(
    ν = νₕ,
    κ = κₕ
);
vertical_diff_closure = VerticalScalarDiffusivity(
    VerticallyImplicitTimeDiscretization(),
    ν = νᵥ,
    κ = κᵥ
);

# Potential AMD closure?

# Turbulence coefficients (AMD)
#const ν = 1.0e-5 # diffusivity of momentum
#const κ = 1.0e-5 # diffusivity of density (or buoyancy)
#ν = νh
#κ = κh

#closure = AnisotropicMinimumDissipation(Cν=ν, Cκ=κ)

closure = (
    horizontal_diff_closure,
    vertical_diff_closure
);

# ================================ #
#           Sponge Layer           #
# ================================ #

# damping timescale (taken from initial time step - τ = 20Δt)
const tau = 5minutes;
const damp_rate = 1/tau;

# NOTE - WILL NEED TO CHANGE ONCE DOMAIN INCREASED
const Lr_sponge = 0.45*Lx;           # sponge layer radial distance
const Lz_sponge = 0.9*Lz;           # sponge layer depth

const Rwidth_sponge = 0.01*Lx;      # sponge layer radial width
const Zwidth_sponge = 0.01*Lz;      # sponge layer z width

@inline mask_2D(x, y, z) = 0.5 .* (tanh.((sqrt.(x.^2 + y.^2) .+ Lr_sponge) ./ Rwidth_sponge) .- tanh.((sqrt.(x.^2 + y.^2) .- Lr_sponge) ./ Rwidth_sponge));
@inline mask_Z(x, y, z) = 0.5 .* (tanh.((z .+ Lz_sponge) ./ Zwidth_sponge) .- tanh.((z .- Lz_sponge) ./ Zwidth_sponge));
@inline mask_net(x, y, z) = 1 - mask_2D(x, y, z) .* mask_Z(x, y, z);

# NOTE - CHANGE ONCE TRACER IMPLEMENTATION COMPLETE
const target_O2 = 0.0;                          # mmol m⁻³
const target_uvw = 0.0;                         # m s⁻¹
@inline target_b(x, y, z, t) = bkgBuoyancy(x, y, z);

uvw_sponge = Relaxation(
    rate = damp_rate,
    mask = mask_net,
    target = target_uvw
);
b_sponge = Relaxation(
    rate = damp_rate,
    mask = mask_net,
    target = target_b
);
O2_sponge = Relaxation(
    rate = damp_rate,
    mask = mask_net,
    target = target_O2
);

# ================================ #
#       Lagrangian Particles       #
# ================================ #

N₀ = 21; # number of particles

x₀ = CUDA.ones(Float64, N₀) .* 125kilometers;
y₀ = CUDA.zeros(Float64, N₀);
z₀ = CuArray(-125.0:-5.0:-225.0);

lagrangian_particles = LagrangianParticles(x=x₀, y=y₀, z=z₀);

# ================================= #
#            Model Setup            #
# ================================= #

model = HydrostaticFreeSurfaceModel(
    grid = grid,
    momentum_advection = WENO(),
    tracer_advection = WENO(),
    coriolis = coriolis,
    buoyancy = BuoyancyTracer(),
    tracers = (:b, :O2),
    particles = lagrangian_particles,
    closure = (horizontal_diff_closure, vertical_diff_closure),
    forcing = (u = uvw_sponge, v = uvw_sponge, w = uvw_sponge, b = b_sponge, O2 = O2_sponge),
    boundary_conditions = (u = U_BCS, v = V_BCS, b = B_BCS, O2 = O2_BCS),
    free_surface = ImplicitFreeSurface()
);

# ================================ #
#      Set Initial Conditions      #
# ================================ #

u, v, w = model.velocities;
b = model.tracers.b;
O2 = model.tracers.O2;

# ϵ parameter to control perturbation. Currently no perturbations
# NOTE - this is problematic for the buoyancy field, because the anomaly is already much weaker
# than the background buoyancy field. Need to make sure perturbations make sense. (talk to Prof. Tandon)
const epsilon = 0.0;
u_perturbation = epsilon .* CUDA.randn(size(u)...) .* U;
v_perturbation = epsilon .* CUDA.randn(size(v)...) .* V;
w_perturbation = Lz/Lx .* epsilon .* CUDA.randn(size(w)...);
b_perturbation = epsilon .* CUDA.randn(size(b)...) .* b_tot;
#O2_perturbation = epsilon .* randn(size(O2)...)

println("Velocity array sizes - U: ", size(u), " V: ", size(v), " W: ", size(w))

Uᵢ = U .+ u_perturbation;
# Uᵢ = CUDA.zeros(size(u)) .+ u_perturbation;
Vᵢ = V .+ v_perturbation;
# Vᵢ = CUDA.zeros(size(v)) .+ v_perturbation;
# NOTE - TODO - FIND OUT WHERE EXTRA DIMENSION IN w IS COMING FROM
Wᵢ = 0.0 .+ w_perturbation;
bᵢ = b_tot.+ b_perturbation;
#O2ᵢ = O2 .+ O2_perturbation

set!(model; b = bᵢ, u = Uᵢ, v = Vᵢ, w = Wᵢ);

# ========================= #
#         Run Setup         #
# ========================= #

simulation = Simulation(model, Δt = 10seconds, stop_time = 100days);
wizard = TimeStepWizard(
    cfl = 0.3,
    max_change = 1.2,
    max_Δt = 5minutes
);
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10));

# ========================= #
#      NetCDF Output        #
# ========================= #

outdir = "../../output/fPlaneWindsPerturb/";

if ~isdir(outdir)
    mkdir(outdir);
end

u, v, w = model.velocities;
b, O2 = model.tracers;
SSH = Field(model.free_surface.η);
ζˣ = Field((∂y(w) - ∂z(v)));
ζʸ = Field((∂z(u) - ∂x(w)));
Qx = Field(((ζˣ * ∂x(b)) / (f₀ * ∂z(b))));
Qy = Field(((ζʸ * ∂y(b)) / (f₀ * ∂z(b))));
Qz = Field(((∂x(v) - ∂y(u) + f₀) / f₀));
Q = Field(Qx + Qy + Qz - 1.0);

extra_outputs = (;
    u=@at((Center, Center, Center), u),
    v=@at((Center, Center, Center), v),
    w=@at((Center, Center, Center), w),
    zeta=@at((Center, Center, Center), ∂x(v)-∂y(u)),
    Q=@at((Center, Center, Center), Q),
    ∇b=@at((Center, Center, Center), sqrt(∂x(b)^2 + ∂y(b)^2)),
    SSH=@at((Center, Center, Nothing), SSH),
)

filename = string(outdir, "output.nc");
simulation.output_writers[:fields] = NetCDFOutputWriter(
    model, merge(model.tracers, extra_outputs),
    overwrite_existing = true,
    filename = filename,
    schedule = TimeInterval(3hours)
);

filename_particles = string(outdir, "particles.nc");
simulation.output_writers[:particles] = NetCDFOutputWriter(
    model, model.particles, overwrite_existing = true,
    filename = filename_particles,
    schedule = TimeInterval(3hours)
);

# ========================= #
#    Progress Messaging     #
# ========================= #

using Printf

function print_progress(simulation)
    u, v, w = simulation.model.velocities

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   iteration(simulation),
                   prettytime(time(simulation)),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   prettytime(simulation.run_wall_time))

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(print_progress, TimeInterval(1hour));

# ========================= #
#         Run Model         #
# ========================= #
# NOTE - will have to call run!(simulation) after including this script
# run!(simulation)
