from copa_map.util import util as ut
from copa_map.util.occ_grid import OccGrid
from copa_map.model.CoPAMapBase import CoPAMapBase
import numpy as np
from os.path import join, exists
import pickle
import matplotlib.colors
from matplotlib import pyplot as plt
import matplotlib.animation as animation

data_folder = join(ut.abs_path(), "data")
opt_params_file = join(data_folder, "grid_atc_opt_params.pickle")

assert exists(opt_params_file), "No saved optimized params found. Run copa_map_atc_train.py before this script."

# Read test data that was already "gridified"
XY = np.genfromtxt(join(data_folder, "grid_atc_50cm_60min_test_xy.csv"), delimiter=",", skip_header=1)
Xt = XY[:, :3]
Yt = XY[:, 3].reshape(-1, 1)
# Load the optimized parameters
with open(opt_params_file, "rb") as f:
    params = pickle.load(f)

# Create a model instance. When providing the saved params, we can directly predict with the optimized model
model = CoPAMapBase(params=params)
# Predict at the testpoints
m_y, stddev_y = model.predict(Xt)

#### PLOTTING
t_bins = np.unique(Xt[:, 2])


def get_values_for_bin(i):
    """Based on the index of the timestamp array, receive the corresponding ground-truth/predicion"""
    t_condition = (Xt[:, 2] == t_bins[i])
    m_y_t = m_y[t_condition]
    s_y_t = stddev_y[t_condition]
    Xt_t_spat = Xt[t_condition][:, :2]
    Yt_t = Yt[t_condition]
    return Xt_t_spat, Yt_t, m_y_t, s_y_t


fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# Load the occupancy map for plotting
occ_map = OccGrid.from_ros_format(path_yaml=join(data_folder, "atc_map.yaml"))
occ_map.plot(axs[0], transparent=True, zorder=2)
occ_map.plot(axs[1], transparent=True, zorder=2)
occ_map.plot(axs[2], transparent=True, zorder=2)

ymin = Xt[:, 1].min() - 2
ymax = Xt[:, 1].max() + 2
xmin = Xt[:, 0].min() - 2
xmax = Xt[:, 0].max() + 2

axs[0].set_ylim(ymin, ymax)
axs[0].set_xlim(xmin, xmax)

m_max = max(np.max(m_y), np.max(Yt))
stdd_max = np.max(stddev_y)
stdd_min = np.min(stddev_y)

Xt_t_spat, Yt_t, m_y_t, s_y_t = get_values_for_bin(0)
# Plot ground truth rate
plot_gt = axs[0].scatter(Xt_t_spat[:, 0], Xt_t_spat[:, 1], c=Yt_t, zorder=1, cmap='jet', s=50.0,
                         norm=matplotlib.colors.Normalize(vmin=0, vmax=m_max), marker="s")
# Plot predictive mean
plot_m = axs[1].scatter(Xt_t_spat[:, 0], Xt_t_spat[:, 1], c=m_y_t, zorder=1, cmap='jet', s=50.0,
                        norm=matplotlib.colors.Normalize(vmin=0, vmax=m_max), marker="s")
# Plot predictive std. deviation
plot_s = axs[2].scatter(Xt_t_spat[:, 0], Xt_t_spat[:, 1], c=s_y_t, zorder=1, cmap='jet', s=50.0,
                        norm=matplotlib.colors.Normalize(vmin=stdd_min, vmax=stdd_max), marker="s")

axs[0].set_title('Ground truth rate y_gt')
axs[1].set_title('Predicted rate m_y')
axs[2].set_title('Predictive std. dev. sigma_y')


def update_scatter(pl, X, c):
    pl.set_offsets(X)
    pl.set_array(c.ravel())


def get_time_str(i):
    ti = t_bins[i]
    ti_day = int((np.floor(ti / 12) + 1))
    ti_h = int(ti % 12 + 9)
    timestr = "t_bin: " + str(ti) + ". Day " + str(ti_day) + ", hour " + str(ti_h) + ":00"
    return timestr


def animate(i):
    Xt_t_spat, Yt_t, m_y_t, s_y_t = get_values_for_bin(i)
    time_txt.set_text(get_time_str(i))
    update_scatter(plot_gt, Xt_t_spat, Yt_t)
    update_scatter(plot_m, Xt_t_spat, m_y_t)
    update_scatter(plot_s, Xt_t_spat, s_y_t)
    return plot_gt, plot_m, plot_s


time_txt = axs[1].text(0.42, 0.9, get_time_str(0), fontsize=14, transform=plt.gcf().transFigure)

# Create time-wise animation
frames = np.arange(0, len(t_bins)).tolist()
anim = animation.FuncAnimation(fig, animate, frames=frames, repeat=False, blit=False, interval=300)
anim.event_source.start()
plt.show()
