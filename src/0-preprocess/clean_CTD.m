% Script to clean N2 data computed from the CTD files
% Cleans NaNs and negative N2 values - also gives the final
% averaged N2 and density profile.

% Author - Parth Sastry (psastry@umassd.edu)

clc; clear; close all;

load("N2_ctd_s37-40.mat");

s37_dens = s37_mini.density;
s37_N2 = s37_mini.N2;
s37_depth = s37_mini.depth;

s38_dens = s38_mini.density;
s38_N2 = s38_mini.N2;
s38_depth = s38_mini.depth;

s39_dens = s39_mini.density;
s39_N2 = s39_mini.N2;
s39_depth = s39_mini.depth;

clear s37_mini s38_mini s39_mini s40_mini;

%% Density/N2 cleanup

% fill NaN with 0
s37_N2(isnan(s37_N2)) = 0;
s38_N2(isnan(s38_N2)) = 0;
s39_N2(isnan(s39_N2)) = 0;

% compute rough MLD, beyond which densities are greater than surface
% density
s37_mld = s37_depth(find(s37_dens < s37_dens(1), 1, 'last'));
s38_mld = s38_depth(find(s38_dens < s38_dens(1), 1, 'last'));
s39_mld = s39_depth(find(s39_dens < s39_dens(1), 1, 'last'));

% hard set densities within approx ML to surface density
s37_dens(s37_depth <= s37_mld) = s37_dens(1);
s38_dens(s38_depth <= s38_mld) = s38_dens(1);
s39_dens(s39_depth <= s39_mld) = s39_dens(1);

% N2 = 0 within ML
s37_N2(s37_depth <= s37_mld) = 0;
s38_N2(s38_depth <= s38_mld) = 0;
s39_N2(s39_depth <= s39_mld) = 0;

% rare occurence of N2 < 0 is set to 0
s37_N2(s37_N2 < 0) = 0;
s38_N2(s38_N2 < 0) = 0;
s39_N2(s39_N2 < 0) = 0;

%% Average/Smoothed N2 - Create averaged N2 profile

% first - clean up N2 to have it across a uniform depth extent for all
init_depth = max([s37_depth(1), s38_depth(1), s39_depth(1)]);
max_depth = 1000;
depth = (init_depth:max_depth).';

avg_N2 = (s37_N2(s37_depth >= init_depth & s37_depth <= max_depth) + ...
    s38_N2(s38_depth >= init_depth & s38_depth <= max_depth) + ...
    s39_N2(s39_depth >= init_depth & s39_depth <= max_depth))/3;

N2 = sgolayfilt(avg_N2, 3, 19);
N2(N2 < 0) = 0;

%% Smoothed background buoyancy profile

% reference surface density (from CTD data)
surf_dens = 1020.5;
g = 9.807;
dens_1000m = (s37_dens(s37_depth == max_depth) + ...
    s38_dens(s38_depth == max_depth) + ...
    s39_dens(s39_depth == max_depth))/3;
surf_b = -g*surf_dens/surf_dens;
b_1000m = -g*dens_1000m/surf_dens;

% integrate to get buoyancy profile
buoyancy = zeros(size(depth));
buoyancy(1) = surf_b;
for i=2:length(depth)
    buoyancy(i) = buoyancy(i-1) - ...
        0.5*(N2(i) + N2(i-1));
end
b_offset = b_1000m - buoyancy(end);
buoyancy = buoyancy + b_offset;

%% Save relevant profiles

clearvars -except N2 buoyancy depth

% extend to surface
depth = -(0:depth(end)).';
buoyancy = vertcat(buoyancy(1)*ones([5,1]), buoyancy);
N2 = vertcat(N2(1)*ones([5,1]), N2);

save("smoothed_N2_bouyancy.mat")

