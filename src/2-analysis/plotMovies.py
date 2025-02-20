import argparse
import os
import xarray as xr

import movieLibrary as ml

parser = argparse.ArgumentParser(
    prog='movies.py',
    description=('Script to create movies of profiles and slices of' +
                 ' relevant variables.')
)
parser.add_argument(
    'output_dir',
    type=str,
    help='Directory containing NC files to be processed.'
)

args = parser.parse_args()
output_dir = args.output_dir

outputDataset = xr.open_dataset(
    output_dir + "output.nc",
    decode_timedelta=True
)
if os.path.exists(output_dir + "particles.nc"):
    particleDataset = xr.open_dataset(
        output_dir + "particles.nc",
        decode_timedelta=True
    )

depth200 = outputDataset.sel(
    zC=-200, method='nearest'
).drop_dims(["zF", "yF", "xF"])
profile = outputDataset.sel(
    yC=0, method='nearest'
).drop_dims(["zF", "yF", "xF"])

ml.plotDepthPV(depth200)
ml.plotProfilepV(profile)

if particleDataset:
    ml.plotParticleTraj(particleDataset)

ml.plotAnomalyProfile(profile)
