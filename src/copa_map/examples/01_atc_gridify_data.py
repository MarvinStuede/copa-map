"""
This script shows how pedestrian detections and robot data can be converted to a spatio-temporal grid

The output data of this script can then be used to train a CoPA-Map model
"""

import pandas as pd
from copa_map.model.Gridifier import Gridifier, GridParams
from copa_map.model.InitInducing import InducingInitializer
from copa_map.util.occ_grid import OccGrid
from copa_map.util import util as ut
from copa_map.util import fov
from os.path import join, exists
from copy import copy
import zipfile
data_folder = join(ut.abs_path(), "data")

csv_names = ["atc_10days_path_pedest_train.csv",
             "atc_10days_path_robot_train.csv",
             "atc_4days_path_pedest_test.csv",
             "atc_4days_path_robot_test.csv"]

# Extract the csv files from the zipped file
all_csv_exist = all([exists(join(data_folder, name)) for name in csv_names])
if not all_csv_exist:
    print("Extracting csv files from zip file...")
    with zipfile.ZipFile(join(data_folder, 'atc_rob_path_ped_detections.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_folder)
a = 1

# Create pd dataframes from csv files
# The csv files with pedestrian data (*_pedest_*) will be converted to a dataframe of the following form:
# [pos_x, pos_y, tidx_bin, t] x [n_det]
# pos_x: First spatial dim of detected pedestrian
# pos_y: Second spatial dim of detected pedestrian
# tidx_bin: The bin where the detection falls into. The data is already pre binned to a bin size of one hour
# t: timestamp of the detection

# The csv files with robot data (*_robot_*) will be converted to a dataframe of the following form:
# [robot_x, robot_y, delta_t, tidx_bin, t] x [n_rob]
# The dataframe then contains the positions of the robot
# robot_x: First spatial dim of the robot
# robot_y: Second spatial dim of the robot
# delta_t: Dwell/rest time of the robot at the corresponding spatial location
# tidx_bin: The bin where the robot position falls into. The data is already pre binned to a bin size of one hour
# t: timestamp of the robot pose
df_data_train = pd.read_csv(join(data_folder, csv_names[0]), index_col=0)
df_rob_train = pd.read_csv(join(data_folder, csv_names[1]), index_col=0)

# Also create for test data, since we only want to test at locations that were visited during training
# The values of the robot's poses correspond to the values from the training data, but the timestamps were adjusted
df_data_test = pd.read_csv(join(data_folder, csv_names[2]), index_col=0)
df_rob_test = pd.read_csv(join(data_folder, csv_names[3]), index_col=0)

# Read the occupancy map to define the location of the grid
occ_map = OccGrid.from_ros_format(path_yaml=join(data_folder, "atc_map.yaml"))

# Overwrite default params of the grid with these params
# For all default params, see class definition of GridParams class
params_grid_train = GridParams(cell_resolution=0.5,
                               origin=occ_map.orig, rotation=occ_map.rotation, width=occ_map.width,
                               height=occ_map.height, rate_min=1e-5, bin_size=3600)
params_grid_test = copy(params_grid_train)

print("Creating grid for training data")
gf_train = Gridifier(occ_map=occ_map, fov=fov.Circle(r=3.5), params=params_grid_train)
gf_train.setup_data(df_data_train, df_rob_train)

print("Creating grid for test data")
gf_test = Gridifier(occ_map=occ_map, fov=fov.Circle(r=3.5), params=params_grid_test, create_gt=True)
gf_test.setup_data(df_data_test, df_rob_test)

# Create the inducing points by clustering
# There are two methods implemented:
# 3D-KMeans: Clustering over the complete input matrix X with targets as weights
# 2D-KMeans: Do separate clustering steps for each time bin. The number of clusters for every bin follows from
#            (number of spatial cells of the bin)/(number of all datapoints) * (number of all inducing points)
print("Clustering for initial inducing point selection...")
init_inducing = InducingInitializer(X=gf_train.get_input_points(), Y_all=gf_train.get_observations(), alpha=0.02)
init_inducing.get_init_inducing(method="2D-KMeans")

path_train = join(data_folder, "grid_atc_50cm_60min_train_xy.csv")
path_test = join(data_folder, "grid_atc_50cm_60min_test_xy.csv")
path_inducing = join(data_folder, "grid_atc_50cm_60min_train_z.csv")

print("Saving training data to " + str(path_train))
gf_train.output_to_text(path_train)

print("Saving inducing point data to " + str(path_inducing))
init_inducing.output_to_text(path_inducing)

print("Saving test data to " + str(path_test))
gf_test.output_to_text(path_test)

print("All data saved. Run 02_copa_map_atc_train.py to train the model.")

plot = True
if plot:
    print("Plotting training data")
    gf_train.plot()
