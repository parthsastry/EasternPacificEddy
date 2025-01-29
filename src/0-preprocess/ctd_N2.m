% Script to use the Processed and Bin-averaged CTD files to compute typical N2 across
% an eddy in Eastern Tropical North Pacific from SR2114 (Chief Sci: Altabet)

% Written by Sid Kerhalkar (skerhalkar@umassd.edu)

% Remember: The data has both up and downcasts Because of the way the CTD and other 
% equipment are configured on the package, the data from the downcast is usually 
% more accurate (from Seabird manual)
%addpath /Users/skerhalkar/Documents/Research/Toolboxes/gsw_matlab_v3_06_16
clc;clear;close all;

s37 = load('/Users/skerhalkar/Documents/Research/ETNP/CTD/processed/S37/S37_E1SR2114_Station37Event1processed_binavg1m.mat');
s38 = load('/Users/skerhalkar/Documents/Research/ETNP/CTD/processed/S38/S38_E3SR2114_Station38Event3processed_binavg1m.mat');
s39 = load('/Users/skerhalkar/Documents/Research/ETNP/CTD/processed/S39/S39_E1SR2114_Station39Event1processed_binavg1m.mat');
s40 = load('/Users/skerhalkar/Documents/Research/ETNP/CTD/processed/S40/S40_E6SR2114_Station40Event6processed_binavg1m.mat');

%% S37 analysis
% Indexing to get the downcast

diff = diff(s37.ctd.depSM);
ind = find(diff<0);
ind = ind(1);
s37_mini.density = s37.ctd.sigma_E00(1:ind)+1000;
s37_mini.depth = s37.ctd.depSM(1:ind);
s37_mini.pressure = s37.ctd.prDM(1:ind);
[s37_mini.N2(2:ind,1),dd]= gsw_Nsquared(s37.ctd.Absolute_Salinity(1:ind)',s37.ctd.Conservative_Temperature(1:ind)',s37.ctd.prDM(1:ind)',mean(s37.ctd.latitude(1:ind),'omitnan'));
s37_mini.N2(1,1)=NaN;
clear ind dd diff 

%% S38 analysis
% Indexing to get the downcast
diff = diff(s38.ctd.depSM);
ind = find(diff<0);
ind = ind(1);
s38_mini.density = s38.ctd.sigma_E00(1:ind)+1000;
s38_mini.depth = s38.ctd.depSM(1:ind);
s38_mini.pressure = s38.ctd.prDM(1:ind);
[s38_mini.N2(2:ind,1),dd]= gsw_Nsquared(s38.ctd.Absolute_Salinity(1:ind)',s38.ctd.Conservative_Temperature(1:ind)',s38.ctd.prDM(1:ind)',mean(s38.ctd.latitude(1:ind),'omitnan'));
s38_mini.N2(1,1)=NaN;
clear ind dd diff

%% s39 analysis
% Indexing to get the downcast
diff = diff(s39.ctd.depSM);
ind = find(diff<0);
ind = ind(1);
s39_mini.density = s39.ctd.sigma_E00(1:ind)+1000;
s39_mini.depth = s39.ctd.depSM(1:ind);
s39_mini.pressure = s39.ctd.prDM(1:ind);
[s39_mini.N2(2:ind,1),dd]= gsw_Nsquared(s39.ctd.Absolute_Salinity(1:ind)',s39.ctd.Conservative_Temperature(1:ind)',s39.ctd.prDM(1:ind)',mean(s39.ctd.latitude(1:ind),'omitnan'));
s39_mini.N2(1,1)=NaN;
clear ind dd diff



%% s40 analysis
% Indexing to get the downcast
diff = diff(s40.ctd.depSM);
ind = find(diff<0);
ind = ind(1);
s40_mini.density = s40.ctd.sigma_E00(1:ind)+1000;
s40_mini.depth = s40.ctd.depSM(1:ind);
s40_mini.pressure = s40.ctd.prDM(1:ind);
[s40_mini.N2(2:ind,1),dd]= gsw_Nsquared(s40.ctd.Absolute_Salinity(1:ind)',s40.ctd.Conservative_Temperature(1:ind)',s40.ctd.prDM(1:ind)',mean(s40.ctd.latitude(1:ind),'omitnan'));
s40_mini.N2(1,1)=NaN;
clear ind dd diffΩ



clear s37 s38 s39 s40
save("N2_ctd_s37-40.mat")

