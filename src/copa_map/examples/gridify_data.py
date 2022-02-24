import numpy as np
import pandas as pd
from copa_map.util import util as ut
from os.path import join, exists
import zipfile
data_folder = join(ut.abs_path(), "data")
csv_names = ["atc_10days_path_pedest_train.csv",
             "atc_10days_path_robot_train.csv",
             "atc_4days_path_pedest_test.csv",
             "atc_4days_path_robot_test.csv"]
all_csv_exist = all([exists(join(data_folder, name)) for name in csv_names])


