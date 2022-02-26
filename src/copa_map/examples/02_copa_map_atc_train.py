"""
This script shows how a CoPA-Map model can be trained, based on the gridified data

The model is saved and predictions/plots can then be made with the 03_copa_map_atc_test.py script
"""

from copa_map.util import util as ut
from copa_map.model.CoPAMapBase import CoPAMapParams, CoPAMapBase
from copa_map.util.occ_grid import OccGrid
import numpy as np
from os.path import join
from copa_map.util.grid_timeseries import GridTimeSeries
import pickle

data_folder = join(ut.abs_path(), "data")

# Read data that was already "gridified"
XY = np.genfromtxt(join(data_folder, "grid_atc_50cm_60min_train_xy.csv"), delimiter=",", skip_header=1)
Z = np.genfromtxt(join(data_folder, "grid_atc_50cm_60min_train_z.csv"), delimiter=",", skip_header=1)

# X: [n x 3] Input data matrix with entries [x1_0, x2_0, t_0; x1_0, x2_1, t_1;...]
# Last row represents the start of each time bin
# In this case, the spatial resolution r_s = 0.5m and temporal resolution tau=60 min
# The timesteps are given as integer numbers where each number represents the number of hours
# Although the data represents multiple days, only the daytime from 9 a.m. to 9 p.m. was considered and gaps between
# days are removed. Therefore t_k from 0 to 11 represents the first day, 12 to 23 the second day and so on
# The same applies to the inducing points Z
X = XY[:, :3]
# Y: [n x 1] Target vector containing the rates of people c/Delta
Y = XY[:, 3].reshape(-1, 1)

# Read the occupancy map to define the location of the grid
occ_map = OccGrid.from_ros_format(path_yaml=join(data_folder, "atc_map.yaml"))

# Now we create a new grid and will do a frequency analysis of the data in the different cells
# Set the grid to the same coordinates as our map
# Resolution is chosen larger than grid for input data of GP.
# This avoids that outliers have a large influence
grid_ts = GridTimeSeries(width=occ_map.width, height=occ_map.height, resolution=5.0,
                         origin=occ_map.orig, rotation=occ_map.rotation,
                         freq_mode="nufft", max_freq_num=10)
# Set the data to the grid
# Our bin size is one hour
bin_size = 3600
grid_ts.set_by_grid_matrices(X=X, Y=Y, bin_size=bin_size)
# Do the frequency analysis for 10 cells, sampled based on their rate
grid_ts.calc_freqs(n_cells=10, sample=True)
# Cluster the results and
psi_arr, sigma2_arr = grid_ts.get_clustered_periods(max_weight=0.95)

# Create Parameters for our model
# Overwrite default params with these params
# For default params, see class definition of CoPAMapParams class
params = CoPAMapParams(opt_max_iters=200,
                       normalize_output=True,
                       normalize_input=True,
                       minibatch_size=1100,
                       use_inducing=True,
                       periods=np.ravel(psi_arr / bin_size).tolist(),
                       period_weights=sigma2_arr.tolist(),
                       likelihood="gaussian_ml",
                       train_inducing=True)

print("Periods (in bin):")
print(params.periods)
print("Weights:")
print(params.period_weights)

model = CoPAMapBase(params=params)
model.learn(X, Y, Z)
opt_params = model.get_model_params()
params_sav_path = join(data_folder, "grid_atc_opt_params.pickle")

with open(params_sav_path, "wb") as f:
    pickle.dump(opt_params, f)

print("Saved optimized parameters to " + params_sav_path)
print("Run 03_copa_map_atc_test.py to visualize results")
