from copa_map.util import util as ut
from copa_map.util.occ_grid import OccGrid
from copa_map.model.CoPAMapBase import CoPAMapBase
import numpy as np
from os.path import join, exists
import pickle


data_folder = join(ut.abs_path(), "data")
opt_params_file = join(data_folder, "grid_atc_opt_params.pickle")

assert exists(opt_params_file), "No saved optimized params found. Run copa_map_atc_train.py before this script."

# Read test data that was already "gridified"
XY = np.genfromtxt(join(data_folder, "grid_atc_50cm_60min_test_xy.csv"), delimiter=",", skip_header=1)
Xt = XY[:, :3]

# Load the optimized parameters
with open(opt_params_file, "rb") as f:
    params = pickle.load(f)

# Create a model instance. When providing the saved params, we can directly predict with the optimized model
model = CoPAMapBase(params=params)
# Predict at the testpoints
m_y, stddev_y = model.predict(Xt)

# Load the occupancy map for plotting
occ_map = OccGrid.from_ros_format(path_yaml=join(data_folder, "atc_map.yaml"))

