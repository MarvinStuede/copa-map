
from copa_map.util import util as ut
from copa_map.model.CoPAMapBase import CoPAMapParams, CoPAMapBase
import numpy as np
from os.path import join
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

# Overwrite default params with these params
# For default params, see class definition of CoPAMapParams class
params = CoPAMapParams(opt_max_iters=200,
                       normalize_output=True,
                       normalize_input=True,
                       minibatch_size=1100,
                       use_inducing=True,
                       likelihood="gaussian_ml",
                       periods=[12.0, 6.0], # From init routine
                       period_weights=[0.92, 0.41], # From init routine
                       train_inducing=True)

model = CoPAMapBase(params=params)
model.learn(X, Y, Z)
opt_params = model.get_model_params()
params_sav_path = join(data_folder, "grid_atc_opt_params.pickle")

with open(params_sav_path, "wb") as f:
    pickle.dump(opt_params, f)

print("Saved  optimized parameters to " + params_sav_path)
print("Run copa_map_atc_test.py to visualize results")